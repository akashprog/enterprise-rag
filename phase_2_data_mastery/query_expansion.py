"""Query expansion on DEV: retrieval only, no answers.

For each DEV question, luna writes three rewrites and one hypothetical passage.
Each text is searched with the same vector and BM25 setup as Phase 2.3, and the
ranked lists are fused with reciprocal rank fusion (k = 60) down to 30 documents.

    0  original question only, the Phase 2.3 fused top 30
    1  original plus the three rewrites
    2  original plus the hypothetical passage
    3  original plus the rewrites and the hypothetical passage

HOLDOUT is not searched. Nothing is answered or judged.

Usage:
    python -m phase_2_data_mastery.query_expansion --dry-run
    python -m phase_2_data_mastery.query_expansion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.ingest_naive import load_docs  # noqa: E402
from phase_2_data_mastery.answer_prompt_v2 import load_split  # noqa: E402
from phase_2_data_mastery.hybrid_candidates import FOCUS, FUSION_DEPTH, load_fused  # noqa: E402
from phase_2_data_mastery.hybrid_check import RRF_K, build_bm25, fuse, gold_found  # noqa: E402
from phase_2_data_mastery.query_small_to_big import DocIndex  # noqa: E402
from phase_2_data_mastery.relevance_filter import (  # noqa: E402
    BILLED_TOKEN_RATIO,
    RELEVANCE_PROMPT_VERSION,
    SEARCH_K,
    Candidate,
    _store_and_index,
    load_candidates,
    passage_text,
    relevance_batches,
    search_candidates,
)
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import dedupe  # noqa: E402
from shared_utils.jev_judge import CachedJev  # noqa: E402
from shared_utils.llm import CachedChat, CachedEmbeddings, run_limited, usage  # noqa: E402

logger = logging.getLogger("phase2.query_expansion")

OUT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "query_expansion_dev.json"
PROMPT_VERSION = "p2-query-expand-v1"
# First cap. The dry-run upper bound assumes this whole allowance is generated.
# A reasoning model can spend the entire cap before it writes the passage; those
# calls are retried once at the higher cap and are not cached.
GEN_MAX_TOKENS = 2000
GEN_MAX_TOKENS_RETRY = 8000
TOP_N = 30
VARIANTS = ("0", "1", "2", "3")
VARIANT_NAMES = {
    "0": "original (Phase 2.3)",
    "1": "original + rewrites",
    "2": "original + hypothetical",
    "3": "original + rewrites + hypothetical",
}

# Rewrites must not add facts. The hypothetical passage is a search probe, so it
# is written as a document that would answer the question.
SYSTEM = (
    "You prepare searches over a company's internal documents.\n"
    "\n"
    "rewrites: exactly three rewrites of the question. Use vocabulary a company "
    "document would likely use: product terms, acronyms spelled out, and synonyms. "
    "Each rewrite must ask the same question. Do not invent names, numbers, or facts.\n"
    "hypothetical_passage: one passage of about 80 words that would answer the "
    "question, written like an internal document such as a wiki page, a ticket, or "
    "a runbook."
)


class Expansion(BaseModel):
    """Three question rewrites and one hypothetical answering passage."""

    rewrites: list[str]
    hypothetical_passage: str


def _clean(parsed: dict | None) -> dict | None:
    """Three non-empty rewrites and a non-empty passage, or None."""
    if not parsed:
        return None
    rewrites = [str(item).strip() for item in parsed.get("rewrites") or []]
    rewrites = [item for item in rewrites if item]
    passage = str(parsed.get("hypothetical_passage") or "").strip()
    if len(rewrites) < 3 or not passage:
        return None
    return {"rewrites": rewrites[:3], "hypothetical_passage": passage, "extra_rewrites": len(rewrites) - 3}


def _messages(question: str) -> list[tuple[str, str]]:
    return [("system", SYSTEM), ("user", f"Question: {question}")]


def dev_questions() -> list:
    """Saved vector candidates for the 210 DEV questions, in split order."""
    dev, holdout = load_split()
    by_id = {q.question_id: q for q in load_candidates()}
    missing = [qid for qid in dev if qid not in by_id]
    if missing or set(dev) & set(holdout):
        raise RuntimeError(f"DEV split does not match the candidate file ({len(missing)} missing).")
    return [by_id[qid] for qid in dev]


def phase23_ids() -> dict[str, list[str]]:
    """Fused top 30 that Phase 2.3 scored, in that fusion order."""
    questions, _ = load_fused()
    return {q.question_id: [c.doc_id for c in q.candidates] for q in questions}


def fuse_many(groups: list[list[str]], *, limit: int) -> list[str]:
    """Reciprocal rank fusion of any number of ranked lists. Same k as Phase 2.3."""
    scores: dict[str, float] = defaultdict(float)
    for ids in groups:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] += 1.0 / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return [doc_id for doc_id, _ in ordered][:limit]


def _ids(rows: list[dict]) -> list[str]:
    return [row["doc_id"] for row in rows]


def verify_original(questions, bm25_rows: dict[str, list[dict]], saved: dict[str, list[str]]) -> None:
    """The original-question fusion must be the list Phase 2.3 already scored."""
    for q in questions:
        vector = [c.doc_id for c in q.candidates][:FUSION_DEPTH]
        got = fuse(vector, _ids(bm25_rows[q.question_id]), limit=TOP_N)
        if got != saved[q.question_id]:
            raise RuntimeError(
                f"{q.question_id}: recomputed fused top 30 does not match Phase 2.3 "
                f"({len(got)} vs {len(saved[q.question_id])} documents)."
            )


def _as_candidate(row: dict) -> Candidate:
    return Candidate(row["doc_id"], row["path"], float(row["score"]), list(row["chunks"]), row["source_type"])


def _best_candidate(doc_id: str, vector_lists: list[list[Candidate]], bm25_lists: list[list[dict]]) -> Candidate:
    """Passage source for one document: a vector hit if any search returned it, else BM25."""
    best: Candidate | None = None
    for cands in vector_lists:
        for cand in cands:
            if cand.doc_id == doc_id and (best is None or cand.score > best.score):
                best = cand
    if best is not None:
        return best
    for rows in bm25_lists:
        for row in rows:
            if row["doc_id"] == doc_id:
                return _as_candidate(row)
    raise RuntimeError(f"No search list contains {doc_id}.")


def titles_by_id() -> dict[str, str]:
    titles: dict[str, str] = {}
    for row in load_docs(None).itertuples():
        titles.setdefault(row.doc_id, row.title)
    return titles


def _hit(ids: list[str], doc_id: str) -> int | None:
    return ids.index(doc_id) + 1 if doc_id in ids else None


def _finder(bundle: dict, doc_id: str) -> dict | None:
    """The expansion text whose own top 50 contains this document, best rank first."""
    best: tuple[int, dict] | None = None
    probes: list[tuple[str, str, list[str], str]] = []
    for i, text in enumerate(bundle["rewrites"]):
        probes.append((f"rewrite {i + 1}", text, bundle["rewrite_vector"][i], "vector"))
        probes.append((f"rewrite {i + 1}", text, bundle["rewrite_bm25"][i], "bm25"))
    probes.append(("hypothetical passage", bundle["hypothetical_passage"], bundle["hypo_vector"], "vector"))
    probes.append(("hypothetical passage", bundle["hypothetical_passage"], bundle["hypo_bm25"], "bm25"))
    for label, text, ids, method in probes:
        rank = _hit(ids, doc_id)
        if rank is None:
            continue
        if best is None or rank < best[0]:
            best = (rank, {"label": label, "text": text, "method": method, "rank": rank})
    return None if best is None else best[1]


def _diverse(rows: list[dict], n: int) -> list[dict]:
    """Up to n rows, different questions, covering as many question types as possible."""
    chosen: list[dict] = []
    seen_q: set[str] = set()
    seen_t: set[str] = set()
    for row in rows:
        if row["question_id"] in seen_q or row["question_type"] in seen_t:
            continue
        chosen.append(row)
        seen_q.add(row["question_id"])
        seen_t.add(row["question_type"])
        if len(chosen) == n:
            return chosen
    for row in rows:
        if row["question_id"] in seen_q:
            continue
        chosen.append(row)
        seen_q.add(row["question_id"])
        if len(chosen) == n:
            break
    return chosen


def _gold_stats(questions, lists: dict[str, dict[str, list[str]]]) -> dict:
    """Gold found, questions rescued, and gold documents each variant drops."""
    base = {qid: set(lists[qid]["0"]) for qid in lists}
    found = {name: gold_found(questions, {qid: lists[qid][name] for qid in lists}, TOP_N) for name in VARIANTS}
    missing_questions = 0
    missing_docs = 0
    rescued_questions = {name: 0 for name in VARIANTS if name != "0"}
    rescued_docs = {name: 0 for name in VARIANTS if name != "0"}
    lost_docs = {name: 0 for name in VARIANTS if name != "0"}
    for q in questions:
        gold = dedupe(q.gold)
        if not gold:
            continue
        had = base[q.question_id]
        absent = [doc_id for doc_id in gold if doc_id not in had]
        if absent:
            missing_questions += 1
            missing_docs += len(absent)
        for name in VARIANTS:
            if name == "0":
                continue
            now = set(lists[q.question_id][name])
            got = [doc_id for doc_id in absent if doc_id in now]
            rescued_docs[name] += len(got)
            if absent and all(doc_id in now for doc_id in gold):
                rescued_questions[name] += 1
            lost_docs[name] += sum(doc_id in had and doc_id not in now for doc_id in gold)
    return {
        "gold_found": found,
        "questions_missing_gold": missing_questions,
        "gold_docs_missing": missing_docs,
        "questions_now_complete": rescued_questions,
        "missing_docs_now_found": rescued_docs,
        "gold_docs_lost": lost_docs,
    }


def _examples(questions, bundles: dict[str, dict], lists: dict[str, dict[str, list[str]]],
              titles: dict[str, str]) -> dict:
    """Five expansion finds and three losses, against the Phase 2.3 list."""
    finds: list[dict] = []
    losses: list[dict] = []
    found_without_probe = 0
    for q in questions:
        bundle = bundles[q.question_id]
        had = set(lists[q.question_id]["0"])
        expanded = {name: set(lists[q.question_id][name]) for name in ("1", "2", "3")}
        for doc_id in dedupe(q.gold):
            if doc_id in had:
                dropped = [name for name in ("1", "2", "3") if doc_id not in expanded[name]]
                if dropped:
                    losses.append({
                        "question_id": q.question_id,
                        "question_type": q.question_type,
                        "question": q.question,
                        "doc_id": doc_id,
                        "title": titles.get(doc_id, ""),
                        "lost_by": dropped,
                    })
                continue
            if doc_id not in expanded["3"] and doc_id not in expanded["1"] and doc_id not in expanded["2"]:
                continue
            probe = _finder(bundle, doc_id)
            if probe is None:
                found_without_probe += 1
                continue
            finds.append({
                "question_id": q.question_id,
                "question_type": q.question_type,
                "question": q.question,
                "doc_id": doc_id,
                "title": titles.get(doc_id, ""),
                "in_variants": [name for name in ("1", "2", "3") if doc_id in expanded[name]],
                **probe,
            })
    finds.sort(key=lambda row: row["rank"])
    # A loss on variant 3 was not saved by either probe. If that list loses
    # fewer than three, fill from the other variants and label which list lost it.
    hard = [row for row in losses if "3" in row["lost_by"]]
    lost = _diverse(hard, 3)
    if len(lost) < 3:
        used = {row["question_id"] for row in lost}
        lost.extend(_diverse([row for row in losses if row["question_id"] not in used], 3 - len(lost)))
    return {
        "found_without_an_expansion_list": found_without_probe,
        "found": _diverse(finds, 5),
        "lost": lost,
    }


def _new_cost(questions, lists, bundles: dict[str, dict], index: DocIndex) -> dict:
    """Jev calls for documents in a variant's top 30 that Phase 2.3 did not score."""
    jev = CachedJev()
    price = settings.model_prices.get(settings.jev_model) or settings.model_prices.get("jev-latest")
    out = {}
    for name in VARIANTS:
        passages = calls = uncached = local_new = 0
        for q in questions:
            base = set(lists[q.question_id]["0"])
            new_ids = [doc_id for doc_id in lists[q.question_id][name] if doc_id not in base]
            if not new_ids:
                continue
            bundle = bundles[q.question_id]
            texts = []
            for doc_id in new_ids:
                cand = _best_candidate(doc_id, bundle["vector_candidates"], bundle["bm25_rows"])
                texts.append(passage_text(cand, index))
            passages += len(texts)
            for batch in relevance_batches(q.question, texts):
                calls += 1
                hit = jev.peek(batch["state"], batch["questions"], prompt_version=RELEVANCE_PROMPT_VERSION)
                if hit is None:
                    uncached += 1
                    local_new += batch["local_tokens"]
        estimate = None if price is None else round(local_new * price.input / 1e6, 4)
        calibrated = None if price is None else round(local_new * BILLED_TOKEN_RATIO * price.input / 1e6, 4)
        out[name] = {
            "new_passages": passages,
            "calls": calls,
            "uncached_calls": uncached,
            "local_tokens": local_new,
            "dry_run_usd": estimate,
            "calibrated_usd": calibrated,
        }
    return out


def _print_report(body: dict) -> None:
    print("\n=== Query expansion, DEV, retrieval only. RRF k=60, top 30. No answers ===")
    gen = body["generation"]
    print(f"dry-run upper bound ${gen['dry_run_usd']}  "
          f"({gen['dry_run_input_tokens']} input, <= {gen['dry_run_output_tokens']} output)")
    if gen.get("cost_usd") is not None:
        print(f"actual generation ${gen['cost_usd']}  "
              f"({gen['input_tokens']} in / {gen['output_tokens']} out), "
              f"embedding ${gen['embedding_usd']}")
    words = body["hypothetical_words"]
    print(f"hypothetical passage words: min {words['min']}, median {words['median']}, "
          f"mean {words['mean']}, max {words['max']}")

    print(f"\n{'type':28s}{'v0':>16s}{'v1 rewrites':>16s}{'v2 passage':>16s}{'v3 both':>16s}")
    found = body["stats"]["gold_found"]
    types = [qtype for qtype in FOCUS if qtype in found["0"]["by_type"] or qtype in found["3"]["by_type"]]

    def cell(block: dict, qtype: str) -> str:
        if qtype == "overall":
            return f"{block['found']}/{block['asked']} ({block['pct']}%)".rjust(16)
        row = block["by_type"].get(qtype)
        if not row:
            return f"{'—':>16s}"
        return f"{row['found']}/{row['asked']} ({row['pct']}%)".rjust(16)

    print(f"{'gold found':28s}{cell(found['0'], 'overall')}{cell(found['1'], 'overall')}"
          f"{cell(found['2'], 'overall')}{cell(found['3'], 'overall')}")
    for qtype in types:
        print(f"{qtype:28s}{cell(found['0'], qtype)}{cell(found['1'], qtype)}"
              f"{cell(found['2'], qtype)}{cell(found['3'], qtype)}")

    stats = body["stats"]
    print(f"\nDEV questions with a gold document outside the Phase 2.3 top 30: "
          f"{stats['questions_missing_gold']}")
    print(f"gold documents missing there: {stats['gold_docs_missing']}")
    for name in ("1", "2", "3"):
        print(f"  {VARIANT_NAMES[name]}: {stats['questions_now_complete'][name]} questions now have "
              f"every gold document, {stats['missing_docs_now_found'][name]} of the missing documents "
              f"are in the top 30, and {stats['gold_docs_lost'][name]} gold documents that "
              f"Phase 2.3 had are gone")

    print("\nNew candidates that Phase 2.3 did not score (calibrated Jev cost, x1.22):")
    for name in VARIANTS:
        row = body["new_candidates"][name]
        print(f"  {VARIANT_NAMES[name]:32s} {row['new_passages']:5d} passages, "
              f"{row['uncached_calls']:4d} new calls, ${row['calibrated_usd']}")

    print("\n=== 5 where expansion put a missing gold document in the top 30 ===")
    for row in body["examples"]["found"]:
        print(f"\n[{row['question_type']}] {row['question_id']}  variants {row['in_variants']}")
        print(f"Q: {row['question']}")
        print(f"Gold: {row['title']}")
        print(f"Found by {row['label']} ({row['method']} rank {row['rank']}): {row['text']}")

    print("\n=== 3 where expansion dropped a gold document Phase 2.3 had ===")
    for row in body["examples"]["lost"]:
        print(f"\n[{row['question_type']}] {row['question_id']}  lost by variants {row['lost_by']}")
        print(f"Q: {row['question']}")
        print(f"Gold: {row['title']}")
    if body["examples"]["found_without_an_expansion_list"]:
        print(f"\nGold documents that entered a top 30 without appearing in any expansion list: "
              f"{body['examples']['found_without_an_expansion_list']}")


async def _generate(questions, *, dry_run: bool) -> tuple[dict[str, dict], dict]:
    llm = CachedChat(settings.expansion_model, max_tokens=GEN_MAX_TOKENS, dry_run=dry_run)
    llm_retry = CachedChat(settings.expansion_model, max_tokens=GEN_MAX_TOKENS_RETRY, dry_run=dry_run)
    extra = 0
    retried = 0

    async def _invoke(chat: CachedChat, question: str, version: str):
        try:
            return await chat.ainvoke(_messages(question), prompt_version=version, schema=Expansion)
        except Exception as exc:
            # Structured output raises when reasoning consumes the whole cap.
            if "length" not in type(exc).__name__.lower() and "length limit" not in str(exc).lower():
                raise
            logger.warning("Expansion hit max_tokens=%d for a question; retrying at %d.",
                           chat.max_tokens, GEN_MAX_TOKENS_RETRY)
            return None

    async def one(q) -> dict:
        nonlocal extra, retried
        res = await _invoke(llm, q.question, PROMPT_VERSION)
        if dry_run:
            return {"question_id": q.question_id}
        cleaned = _clean(res.parsed) if res is not None else None
        if cleaned is None:
            retried += 1
            res = await _invoke(llm_retry, q.question, PROMPT_VERSION)
            cleaned = _clean(res.parsed) if res is not None else None
        if cleaned is None:
            raise RuntimeError(f"{q.question_id}: expansion did not return three rewrites and a passage.")
        extra += cleaned["extra_rewrites"]
        return {
            "question_id": q.question_id,
            "rewrites": cleaned["rewrites"],
            "hypothetical_passage": cleaned["hypothetical_passage"],
            "hypothetical_words": len(cleaned["hypothetical_passage"].split()),
        }

    usage.reset()
    rows = await run_limited(questions, one, limit=5, desc="query expansion")
    report = usage.report()
    model = next(iter(report["models"].values()))
    if dry_run:
        cost = {
            "dry_run_usd": report.get("total_estimated_cost_usd"),
            "dry_run_input_tokens": model["estimated_input_tokens"],
            "dry_run_output_tokens": model["estimated_output_tokens"],
        }
        print("\n=== query expansion dry-run upper bound ===")
        print(json.dumps(report, indent=2))
        return {}, cost
    by_id = {row["question_id"]: row for row in rows}
    words = [row["hypothetical_words"] for row in rows]
    cost = {
        "dry_run_usd": None,
        "cost_usd": report.get("total_cost_usd"),
        "input_tokens": model["input_tokens"],
        "output_tokens": model["output_tokens"],
        "calls": model["calls"],
        "cached_calls": model["cached_calls"],
        "extra_rewrites_dropped": extra,
        "retries": retried,
        "hypothetical_words": {
            "min": min(words), "median": statistics.median(words),
            "mean": round(statistics.mean(words), 1), "max": max(words),
        },
    }
    print("\n=== query expansion generation cost ===")
    print(json.dumps(report, indent=2))
    return by_id, cost


def _search(questions, expansions: dict[str, dict], bm25_index, meta, bm25_original: dict[str, list[dict]]):
    """Vector and BM25 lists for every expansion text. Original vector is the saved top 50."""
    store, doc_index = _store_and_index()
    texts = []
    for row in expansions.values():
        texts.extend(row["rewrites"])
        texts.append(row["hypothetical_passage"])
    embed = CachedEmbeddings()
    estimate = embed.estimate_cost(texts)
    print("\n=== embedding the rewrites and hypothetical passages ===")
    print(json.dumps(estimate, indent=2))
    usage.reset()
    embed.embed_documents(texts)
    embed_cost = usage.report()

    bundles: dict[str, dict] = {}
    lists: dict[str, dict[str, list[str]]] = {}
    for n, q in enumerate(questions, start=1):
        row = expansions[q.question_id]
        original_vector = list(q.candidates[:FUSION_DEPTH])
        rewrite_vector = [search_candidates(text, store, doc_index, k=SEARCH_K)[:FUSION_DEPTH] for text in row["rewrites"]]
        hypo_vector = search_candidates(row["hypothetical_passage"], store, doc_index, k=SEARCH_K)[:FUSION_DEPTH]
        rewrite_bm25 = [bm25_index.ranked_documents(text, meta, limit=FUSION_DEPTH) for text in row["rewrites"]]
        hypo_bm25 = bm25_index.ranked_documents(row["hypothetical_passage"], meta, limit=FUSION_DEPTH)
        original_bm25 = bm25_original[q.question_id]
        groups = {
            "0": [ _ids_from_cands(original_vector), _ids(original_bm25) ],
            "1": [_ids_from_cands(original_vector), _ids(original_bm25)]
                 + [_ids_from_cands(c) for c in rewrite_vector] + [_ids(r) for r in rewrite_bm25],
            "2": [_ids_from_cands(original_vector), _ids(original_bm25),
                  _ids_from_cands(hypo_vector), _ids(hypo_bm25)],
            "3": [_ids_from_cands(original_vector), _ids(original_bm25)]
                 + [_ids_from_cands(c) for c in rewrite_vector] + [_ids(r) for r in rewrite_bm25]
                 + [_ids_from_cands(hypo_vector), _ids(hypo_bm25)],
        }
        lists[q.question_id] = {name: fuse_many(groups[name], limit=TOP_N) if name != "0"
                                else fuse(_ids_from_cands(original_vector), _ids(original_bm25), limit=TOP_N)
                                for name in VARIANTS}
        bundles[q.question_id] = {
            "rewrites": row["rewrites"],
            "hypothetical_passage": row["hypothetical_passage"],
            "rewrite_vector": [[c.doc_id for c in cands] for cands in rewrite_vector],
            "rewrite_bm25": [_ids(r) for r in rewrite_bm25],
            "hypo_vector": [c.doc_id for c in hypo_vector],
            "hypo_bm25": _ids(hypo_bm25),
            "vector_candidates": [original_vector, *rewrite_vector, hypo_vector],
            "bm25_rows": [original_bm25, *rewrite_bm25, hypo_bm25],
        }
        if n % 30 == 0 or n == len(questions):
            logger.info("Searched %d / %d DEV questions.", n, len(questions))
    return bundles, lists, doc_index, embed_cost


def _ids_from_cands(cands: list[Candidate]) -> list[str]:
    return [c.doc_id for c in cands]


def _persist(body: dict) -> None:
    """Write the report. Search objects are not serialisable, so they are dropped."""
    slim_questions = []
    for row in body["questions"]:
        slim_questions.append({k: v for k, v in row.items() if k not in ("vector_candidates", "bm25_rows")})
    payload = {**body, "questions": slim_questions}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


async def main_async(dry_run: bool) -> int:
    questions = dev_questions()
    if len(questions) != 210:
        raise RuntimeError(f"Expected 210 DEV questions, got {len(questions)}.")
    _, dry_cost = await _generate(questions, dry_run=True)
    if dry_run:
        return 0

    logger.info("Checking that the original fusion matches the Phase 2.3 list.")
    bm25_index, meta, n_chunks = build_bm25()
    logger.info("BM25 index over %d chunks.", n_chunks)
    bm25_original = {q.question_id: bm25_index.ranked_documents(q.question, meta, limit=FUSION_DEPTH)
                     for q in questions}
    saved = phase23_ids()
    verify_original(questions, bm25_original, saved)

    expansions, gen_cost = await _generate(questions, dry_run=False)
    gen_cost["dry_run_usd"] = dry_cost["dry_run_usd"]
    gen_cost["dry_run_input_tokens"] = dry_cost["dry_run_input_tokens"]
    gen_cost["dry_run_output_tokens"] = dry_cost["dry_run_output_tokens"]
    # Keep the probes even if the search below fails, so a rerun does not pay again.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    (OUT_PATH.parent / "query_expansion_texts.json").write_text(json.dumps(
        {"prompt_version": PROMPT_VERSION, "system": SYSTEM, "questions": list(expansions.values())},
        indent=2, ensure_ascii=False))

    bundles, lists, doc_index, embed_cost = _search(questions, expansions, bm25_index, meta, bm25_original)
    verify_original(questions, bm25_original, saved)
    for q in questions:
        if lists[q.question_id]["0"] != saved[q.question_id]:
            raise RuntimeError(f"{q.question_id}: variant 0 drifted from Phase 2.3.")

    gen_cost["embedding_usd"] = embed_cost.get("total_cost_usd")
    words = [row["hypothetical_words"] for row in expansions.values()]
    titles = titles_by_id()
    stats = _gold_stats(questions, lists)
    examples = _examples(questions, bundles, lists, titles)
    new_cost = _new_cost(questions, lists, bundles, doc_index)
    body = {
        "prompt_version": PROMPT_VERSION,
        "rrf_k": RRF_K,
        "fusion_depth": FUSION_DEPTH,
        "top_n": TOP_N,
        "questions_n": len(questions),
        "generation": gen_cost,
        "hypothetical_words": {
            "min": min(words), "median": statistics.median(words),
            "mean": round(statistics.mean(words), 1), "max": max(words),
        },
        "stats": stats,
        "new_candidates": new_cost,
        "examples": examples,
        "questions": [
            {
                "question_id": q.question_id,
                "lists": lists[q.question_id],
                "rewrites": bundles[q.question_id]["rewrites"],
                "hypothetical_passage": bundles[q.question_id]["hypothetical_passage"],
                "rewrite_vector": bundles[q.question_id]["rewrite_vector"],
                "rewrite_bm25": bundles[q.question_id]["rewrite_bm25"],
                "hypo_vector": bundles[q.question_id]["hypo_vector"],
                "hypo_bm25": bundles[q.question_id]["hypo_bm25"],
            }
            for q in questions
        ],
    }
    _persist(body)
    _print_report(body)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEV query expansion, retrieval check only.")
    parser.add_argument("--dry-run", action="store_true", help="Price the luna calls and stop.")
    args = parser.parse_args(argv)
    setup_logging()
    return asyncio.run(main_async(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
