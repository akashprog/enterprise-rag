"""Step 2.5: search with a hypothetical passage, then answer as the baseline does.

Kept on disk. Not the Phase 2 base pipeline. HOLDOUT matched Phase 2.3 on the
full focus set and was lower on HOLDOUT itself, so the base pipeline is
Phase 2.3 plus the floor line.

The hypothetical passage is a search probe only. It is not a document, and it
is not written into the Jev state or the answer prompt. `assert_probe_excluded`
checks the state and the prompt that will actually be sent.

This module ran DEV and the info_not_found guard. HOLDOUT confirmation is
`holdout_confirm`.

Usage:
    python -m phase_2_data_mastery.expand_answer --dry-run
    python -m phase_2_data_mastery.expand_answer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.ingest_naive import load_docs  # noqa: E402
from phase_1_baseline.query_naive import SYSTEM_PROMPT  # noqa: E402
from phase_2_data_mastery.answer_prompt_v2 import PHASE23_DIR, load_split, prepare  # noqa: E402
from phase_2_data_mastery.floor_line import (  # noqa: E402
    OUT_DIR as FLOOR_DIR,
    PROMPT_VERSION as FLOOR_VERSION,
    user_prompt,
)
from phase_2_data_mastery.hybrid_candidates import (  # noqa: E402
    ANSWER_PROMPT_VERSION,
    FOCUS,
    FUSION_DEPTH,
    GUARD,
    load_fused,
)
from phase_2_data_mastery.hybrid_check import build_bm25, fuse  # noqa: E402
from phase_2_data_mastery.query_expansion import (  # noqa: E402
    OUT_PATH as EXPANSION_PATH,
    TOP_N,
    _as_candidate,
    _generate,
    fuse_many,
)
from phase_2_data_mastery.query_small_to_big import DocIndex, messages  # noqa: E402
from phase_2_data_mastery.relevance_filter import (  # noqa: E402
    BILLED_TOKEN_RATIO,
    DEFAULT_BUDGET,
    SEARCH_K,
    Candidate,
    _store_and_index,
    attach_passages,
    load_candidates,
    passage_text,
    relevance_batches,
    score_passages,
    search_candidates,
    select_by_probability,
    summarise_kept,
)
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import QuestionScore, aggregate, dedupe  # noqa: E402
from shared_utils.llm import CachedChat, count_tokens, truncate_to_tokens, usage  # noqa: E402
from shared_utils.runner import _run_phase  # noqa: E402

logger = logging.getLogger("phase2.expand_answer")

OUT_DIR = settings.paths.results_dir / "phase_2" / "p2_expand_passage_dev"
GUARD_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "query_expansion_guard.json"
REPORT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "expand_passage_dev.json"
SCORES_PATH = OUT_DIR / "scores.jsonl"
_SCORE_FIELDS = {f.name for f in fields(QuestionScore)}
# DEV generation wall clock from the completed retrieval run (210 calls, concurrency 5).
DEV_EXPANSION_SECONDS = 284.0


def assert_probe_excluded(probe: str, question: str, passages: list[str], where: str) -> None:
    """Fail if the search probe would be visible to the filter or the answer model.

    The Jev state is the question plus these passage texts. The answer prompt
    is the same passages. The probe is allowed to share words with them; it is
    not allowed to be one of the strings they contain.
    """
    if not probe or not probe.strip():
        raise RuntimeError(f"Missing hypothetical passage before {where}.")
    blob_passages = list(passages)
    if probe == question or any(probe in text for text in blob_passages):
        raise RuntimeError(f"Hypothetical passage would be visible to {where}.")
    # The state Jev will receive, built by the same helper the scorer uses.
    for batch in relevance_batches(question, passages):
        if probe in json.dumps(batch["state"]):
            raise RuntimeError(f"Hypothetical passage is inside the Jev state ({where}).")


def _score(raw: dict) -> QuestionScore:
    return QuestionScore(**{k: raw[k] for k in _SCORE_FIELDS if k in raw})


def _load_eval(path: Path) -> dict[str, QuestionScore]:
    return {row.question_id: row for row in (_score(json.loads(line)) for line in path.open(encoding="utf-8"))}


def _titles() -> dict[str, str]:
    titles: dict[str, str] = {}
    for row in load_docs(None).itertuples():
        titles.setdefault(row.doc_id, row.title)
    return titles


def _dev_and_guard() -> tuple[list, list, list[str], list[str]]:
    """Vector-search questions for DEV and the guard. HOLDOUT is not included."""
    dev_ids, hold_ids = load_split()
    by_id = {q.question_id: q for q in load_candidates()}
    fused, _ = load_fused()
    guard_ids = [q.question_id for q in fused if q.question_type in GUARD]
    if len(dev_ids) != 210 or len(guard_ids) != 20:
        raise RuntimeError(f"Expected 210 DEV and 20 guard, got {len(dev_ids)} and {len(guard_ids)}.")
    if set(dev_ids) & set(hold_ids) or set(guard_ids) & set(hold_ids) or set(dev_ids) & set(guard_ids):
        raise RuntimeError("DEV, HOLDOUT, and the guard are not disjoint.")
    missing = [qid for qid in dev_ids + guard_ids if qid not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} questions are missing from the candidate file.")
    return [by_id[qid] for qid in dev_ids], [by_id[qid] for qid in guard_ids], dev_ids, guard_ids


def _saved_dev() -> dict[str, dict]:
    raw = json.loads(EXPANSION_PATH.read_text())
    return {row["question_id"]: row for row in raw["questions"]}


def _dev_generation_cost() -> dict:
    raw = json.loads(EXPANSION_PATH.read_text())["generation"]
    return {
        "cost_usd": raw.get("cost_usd"),
        "input_tokens": raw.get("input_tokens"),
        "output_tokens": raw.get("output_tokens"),
        "calls": raw.get("calls"),
        "cached_calls": raw.get("cached_calls"),
        "seconds": DEV_EXPANSION_SECONDS,
    }


def _materialize(q, fused_ids: list[str], pools: list[list[Candidate]], bm25_pools: list[list[dict]]) -> list[Candidate]:
    """One candidate per fused doc id. A vector hit keeps its own chunks."""
    by_id: dict[str, Candidate] = {}
    for pool in pools:
        for cand in pool:
            by_id.setdefault(cand.doc_id, cand)
    for rows in bm25_pools:
        for row in rows:
            by_id.setdefault(row["doc_id"], _as_candidate(row))
    missing = [doc_id for doc_id in fused_ids if doc_id not in by_id]
    if missing:
        raise RuntimeError(f"{q.question_id} fused list has no passage source for {missing[:3]}.")
    return [by_id[doc_id] for doc_id in fused_ids]


def _build(questions, probes: dict[str, str], saved_lists: dict[str, list[str]] | None) -> list:
    """Fused top 30 of the original question and the hypothetical passage.

    `saved_lists` is the DEV check's variant 2. A rebuild that disagrees with it
    is a different experiment, so it stops before any scoring.
    """
    logger.info("Building the fused top 30 for %d questions.", len(questions))
    bm25_index, meta, n_chunks = build_bm25()
    logger.info("BM25 index over %d chunks.", n_chunks)
    store, doc_index = _store_and_index()
    fused_rows, _ = load_fused()
    phase23 = {q.question_id: [c.doc_id for c in q.candidates] for q in fused_rows}
    out = []
    for n, q in enumerate(questions, start=1):
        probe = probes[q.question_id]
        vector = list(q.candidates[:FUSION_DEPTH])
        original_bm25 = bm25_index.ranked_documents(q.question, meta, limit=FUSION_DEPTH)
        original_ids = fuse(
            [c.doc_id for c in vector],
            [row["doc_id"] for row in original_bm25],
            limit=TOP_N,
        )
        if original_ids != phase23[q.question_id]:
            raise RuntimeError(f"{q.question_id}: original fusion does not match Phase 2.3.")
        hypo_vector = search_candidates(probe, store, doc_index, k=SEARCH_K)[:FUSION_DEPTH]
        hypo_bm25 = bm25_index.ranked_documents(probe, meta, limit=FUSION_DEPTH)
        fused_ids = fuse_many([
            [c.doc_id for c in vector],
            [row["doc_id"] for row in original_bm25],
            [c.doc_id for c in hypo_vector],
            [row["doc_id"] for row in hypo_bm25],
        ], limit=TOP_N)
        if saved_lists is not None and q.question_id in saved_lists and fused_ids != saved_lists[q.question_id]:
            raise RuntimeError(f"{q.question_id}: rebuilt '+ passage' list does not match the check.")
        cands = _materialize(q, fused_ids, [vector, hypo_vector], [original_bm25, hypo_bm25])
        texts = [passage_text(c, doc_index) for c in cands]
        assert_probe_excluded(probe, q.question, texts, "the Jev filter")
        out.append(type(q)(q.question_id, q.question_type, q.question, list(q.gold), cands))
        if n % 30 == 0 or n == len(questions):
            logger.info("Built %d / %d candidate lists.", n, len(questions))
    attach_passages([c for q in out for c in q.candidates], doc_index)
    return out


def _pack_upper(q, index: DocIndex) -> str:
    """Every passage that fits in the budget, before the relevance cutoff."""
    dummy = [1.0] * len(q.candidates)
    sent = select_by_probability(q.candidates, dummy, threshold=0.0, budget=DEFAULT_BUDGET, floor=1)
    texts = []
    for cand, mode, n_tokens in sent:
        text = passage_text(cand, index)
        if mode.endswith("+cut"):
            text = truncate_to_tokens(text, n_tokens)
        texts.append(text)
    return "\n\n".join(texts)


def _answer_tokens(questions, probes: dict[str, str], index: DocIndex) -> dict:
    """Upper bound: full budget of passages, floor line included, full output cap."""
    in_tok = 0
    for q in questions:
        context = _pack_upper(q, index)
        assert_probe_excluded(probes[q.question_id], q.question, [context], "the answer dry-run")
        messages_ = [("system", SYSTEM_PROMPT), ("user", user_prompt(q.question, context))]
        in_tok += sum(count_tokens(text) for _, text in messages_)
    out_tok = len(questions) * settings.answer_max_tokens
    return {
        "questions": len(questions),
        "input_tokens": in_tok,
        "output_tokens_cap": out_tok,
        "usd": usage.price(settings.answer_model, in_tok, out_tok),
    }


async def _guard_probes(guard_qs, *, generate: bool) -> tuple[dict[str, str] | None, dict, float | None]:
    """Hypothetical passages for the 20 guard questions. DEV passages are already saved."""
    if GUARD_PATH.is_file():
        raw = json.loads(GUARD_PATH.read_text())
        rows = {row["question_id"]: row["hypothetical_passage"] for row in raw["questions"]}
        if set(rows) == {q.question_id for q in guard_qs}:
            logger.info("Guard hypothetical passages loaded from disk.")
            return rows, raw.get("cost") or {}, raw.get("seconds")
    _, dry = await _generate(guard_qs, dry_run=True)
    if not generate:
        return None, dry, None
    started = time.perf_counter()
    generated, cost = await _generate(guard_qs, dry_run=False)
    elapsed = round(time.perf_counter() - started, 1)
    payload = {
        "questions": [
            {"question_id": qid, "hypothetical_passage": row["hypothetical_passage"],
             "rewrites": row["rewrites"]}
            for qid, row in generated.items()
        ],
        "cost": cost,
        "seconds": elapsed,
    }
    GUARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUARD_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return {qid: row["hypothetical_passage"] for qid, row in generated.items()}, cost, elapsed


def _save_scores(questions, probs: dict[str, list[float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SCORES_PATH.open("w", encoding="utf-8") as handle:
        for q in questions:
            handle.write(json.dumps({
                "question_id": q.question_id,
                "doc_ids": [c.doc_id for c in q.candidates],
                "probs": probs[q.question_id],
            }) + "\n")


def _load_scores(questions) -> dict[str, list[float]] | None:
    if not SCORES_PATH.is_file():
        return None
    raw = {row["question_id"]: row for row in map(json.loads, SCORES_PATH.open())}
    if set(raw) != {q.question_id for q in questions}:
        return None
    probs = {}
    for q in questions:
        row = raw[q.question_id]
        if row["doc_ids"] != [c.doc_id for c in q.candidates]:
            return None
        probs[q.question_id] = row["probs"]
    return probs


def _prompt_tokens(item: dict) -> int:
    context = "\n\n".join(p["text"] for p in item["passages"])
    if item["floor"]:
        pairs = [("system", SYSTEM_PROMPT), ("user", user_prompt(item["question"], context))]
    else:
        pairs = messages(item["question"], context)
    return sum(count_tokens(text) for _, text in pairs)


def _retrieval(questions, prepared: dict[str, dict]) -> dict:
    kept = {qid: item["doc_ids"] for qid, item in prepared.items()}
    stats = summarise_kept(questions, kept)
    buckets: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        buckets[q.question_type].append(_prompt_tokens(prepared[q.question_id]))
        buckets["ALL"].append(_prompt_tokens(prepared[q.question_id]))
    stats["overall"]["prompt_tokens_avg"] = round(statistics.mean(buckets["ALL"]), 1)
    for qtype, block in stats["by_type"].items():
        block["prompt_tokens_avg"] = round(statistics.mean(buckets[qtype]), 1)
    return stats


def _baseline_answers(dev_ids: list[str], guard_ids: list[str]) -> dict[str, QuestionScore]:
    """Phase 2.3 verdicts, with the floor-line re-answers in place of those 22."""
    phase23 = _load_eval(PHASE23_DIR / "jev" / "answers_eval.jsonl")
    floor = _load_eval(FLOOR_DIR / "jev" / "answers_eval.jsonl")
    merged = dict(phase23)
    merged.update(floor)
    wanted = set(dev_ids) | set(guard_ids)
    missing = wanted - set(merged)
    if missing:
        raise RuntimeError(f"Baseline is missing {len(missing)} verdicts.")
    return {qid: merged[qid] for qid in wanted}


def _flips(before: dict[str, QuestionScore], after: dict[str, QuestionScore], ids: list[str],
           old_docs: dict[str, list[str]], new_docs: dict[str, list[str]],
           gold: dict[str, list[str]], titles: dict[str, str]) -> list[dict]:
    rows = []
    for qid in ids:
        old, new = bool(before[qid].correct), bool(after[qid].correct)
        if old == new:
            continue
        old_set, new_set = set(old_docs[qid]), set(new_docs[qid])
        gained = [doc_id for doc_id in dedupe(gold[qid]) if doc_id in new_set and doc_id not in old_set]
        lost = [doc_id for doc_id in dedupe(gold[qid]) if doc_id in old_set and doc_id not in new_set]
        rows.append({
            "question_id": qid,
            "question_type": after[qid].question_type,
            "direction": "wrong_to_right" if new else "right_to_wrong",
            "documents_changed": old_set != new_set,
            "gold_gained": [{"doc_id": doc_id, "title": titles.get(doc_id, "")} for doc_id in gained],
            "gold_lost": [{"doc_id": doc_id, "title": titles.get(doc_id, "")} for doc_id in lost],
        })
    return rows


def _gained_gold(saved: dict[str, dict], questions, new_docs: dict[str, list[str]],
                 after: dict[str, QuestionScore], before: dict[str, QuestionScore],
                 top_scores: dict[str, float], titles: dict[str, str]) -> list[dict]:
    """The DEV questions whose '+ passage' list contains a gold doc Phase 2.3's list did not."""
    rows = []
    by_id = {q.question_id: q for q in questions}
    for qid, row in saved.items():
        q = by_id.get(qid)
        if q is None:
            continue
        had, now = set(row["lists"]["0"]), set(row["lists"]["2"])
        gained = [doc_id for doc_id in dedupe(q.gold) if doc_id not in had and doc_id in now]
        if not gained:
            continue
        sent = set(new_docs[qid])
        rows.append({
            "question_id": qid,
            "question_type": q.question_type,
            "question": q.question,
            "gold": [{"doc_id": doc_id, "title": titles.get(doc_id, ""), "kept": doc_id in sent} for doc_id in gained],
            "correct_before": bool(before[qid].correct),
            "correct_after": bool(after[qid].correct),
            "phase23_top_score": top_scores[qid],
        })
    return rows


def _top_scores() -> dict[str, float]:
    """Highest Phase 2.3 Jev score on each question's fused top 30."""
    _, probs = load_fused()
    return {qid: max(values) for qid, values in probs.items()}


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    def at(p: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))], 4)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p25": at(0.25),
        "median": round(statistics.median(ordered), 4),
        "p75": at(0.75),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.mean(ordered), 4),
    }


def _runtime_signal(dev_ids: list[str], before: dict[str, QuestionScore], old_docs: dict[str, list[str]],
                    gold: dict[str, list[str]], flips: list[dict]) -> dict:
    """Top Phase 2.3 score by whether the baseline prompt held the gold and the answer was right."""
    top = _top_scores()
    groups: dict[str, list[float]] = {"a_gold_in_correct": [], "b_gold_in_wrong": [], "c_gold_missing": []}
    no_gold = []
    correct_but_missing = 0
    for qid in dev_ids:
        needed = dedupe(gold[qid])
        if not needed:
            no_gold.append(qid)
            continue
        present = all(doc_id in set(old_docs[qid]) for doc_id in needed)
        if not present:
            groups["c_gold_missing"].append(top[qid])
            if before[qid].correct:
                correct_but_missing += 1
        elif before[qid].correct:
            groups["a_gold_in_correct"].append(top[qid])
        else:
            groups["b_gold_in_wrong"].append(top[qid])
    thresholds = {}
    for cutoff in (0.7, 0.8, 0.9, 0.95):
        below = [qid for qid in dev_ids if qid not in no_gold and top[qid] < cutoff]
        in_c = [
            qid for qid in below
            if not all(doc_id in set(old_docs[qid]) for doc_id in dedupe(gold[qid]))
        ]
        thresholds[str(cutoff)] = {"below": len(below), "group_c": len(in_c)}
    fixed = []
    for row in flips:
        if row["direction"] != "wrong_to_right":
            continue
        fixed.append({
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "phase23_top_score": round(top[row["question_id"]], 4),
        })
    return {
        "excluded_no_gold": no_gold,
        "correct_with_gold_missing": correct_but_missing,
        "distribution": {name: _dist(values) for name, values in groups.items()},
        "thresholds": thresholds,
        "fixed_top_scores": fixed,
    }


def _expansion_cost(dev_cost: dict, guard_cost: dict | None, guard_seconds: float | None, n_dev: int, n_guard: int) -> dict:
    dev_each = None if not dev_cost.get("cost_usd") else round(dev_cost["cost_usd"] / n_dev, 6)
    dev_time = round(dev_cost["seconds"] / n_dev, 2)
    guard_each = None
    guard_time = None
    if guard_cost and guard_cost.get("cost_usd") is not None and n_guard:
        guard_each = round(guard_cost["cost_usd"] / n_guard, 6)
    if guard_seconds is not None and n_guard:
        guard_time = round(guard_seconds / n_guard, 2)
    return {
        "dev": {"questions": n_dev, "total_usd": dev_cost.get("cost_usd"), "usd_per_question": dev_each,
                "wall_seconds": dev_cost.get("seconds"), "seconds_per_question": dev_time,
                "note": "One call also wrote three rewrites this step does not search with. "
                        "Wall clock is the completed DEV run at concurrency 5."},
        "guard": {"questions": n_guard, "total_usd": None if not guard_cost else guard_cost.get("cost_usd"),
                  "usd_per_question": guard_each, "wall_seconds": guard_seconds,
                  "seconds_per_question": guard_time},
    }


def _print_retrieval(label: str, stats: dict) -> None:
    print(f"\n{label}")
    print(f"{'type':28s}{'recall':>8s}{'invalid':>8s}{'docs':>8s}{'tokens':>8s}")
    overall = stats["overall"]
    print(f"{'ALL':28s}{overall['recall10']:8.2f}{overall['invalid']:8.2f}"
          f"{overall['avg_docs']:8.2f}{overall['prompt_tokens_avg']:8.1f}")
    for qtype in FOCUS:
        block = stats["by_type"].get(qtype)
        if not block:
            continue
        recall = "       —" if block["recall10"] is None else f"{block['recall10']:8.2f}"
        invalid = "       —" if block["invalid"] is None else f"{block['invalid']:8.2f}"
        print(f"{qtype:28s}{recall}{invalid}{block['avg_docs']:8.2f}{block['prompt_tokens_avg']:8.1f}")


def _print_answers(label: str, block: dict) -> None:
    print(f"{label:28s}{block['overall_score']:8.2f}{block['correctness']:8.2f}"
          f"{block['completeness']:8.2f}{block['avg_answer_words']:8.1f}")


def _print_report(body: dict) -> None:
    print("\n=== Step 2.5  hypothetical passage, fused top 30. Baseline is Phase 2.3 + floor line. Jev ===")
    costs = body["costs"]
    print(f"Jev scoring ${costs['jev_usd']}   answering ${costs['answer_usd']}   judging ${costs['judge_usd']}")
    exp = body["expansion_per_question"]
    print(f"expansion DEV ${exp['dev']['usd_per_question']} and {exp['dev']['seconds_per_question']}s per question")
    guard = exp["guard"]
    if guard["usd_per_question"] is not None:
        print(f"expansion guard ${guard['usd_per_question']} and {guard['seconds_per_question']}s per question")
    _print_retrieval("DEV retrieval, baseline", body["retrieval"]["baseline"])
    _print_retrieval("DEV retrieval, + passage", body["retrieval"]["expanded"])
    print(f"\n{'':28s}{'overall':>8s}{'correct':>8s}{'complete':>8s}{'words':>8s}")
    dev = body["dev"]
    _print_answers("DEV baseline", dev["baseline"])
    _print_answers("DEV + passage", dev["expanded"])
    print(f"flips: {dev['wrong_to_right']} wrong->right, {dev['right_to_wrong']} right->wrong")
    print(f"\n{'type':28s}{'base':>8s}{'new':>8s}{'w->r':>6s}{'r->w':>6s}")
    for qtype in FOCUS:
        old = dev["baseline_by_type"].get(qtype, {})
        new = dev["expanded_by_type"].get(qtype, {})
        flips = dev["flips_by_type"].get(qtype, {})
        print(f"{qtype:28s}{old.get('overall_score', 0):8.2f}{new.get('overall_score', 0):8.2f}"
              f"{flips.get('wrong_to_right', 0):6d}{flips.get('right_to_wrong', 0):6d}")
    guard_block = body["info_not_found"]
    print("\ninfo_not_found, against 100")
    _print_answers("baseline", guard_block["baseline"])
    _print_answers("+ passage", guard_block["expanded"])
    print(f"flips: {guard_block['wrong_to_right']} wrong->right, {guard_block['right_to_wrong']} right->wrong")
    print("\nFlips (set of documents, gold gained or lost)")
    for row in body["flips"]:
        gained = ", ".join(item["title"] or item["doc_id"] for item in row["gold_gained"]) or "none"
        lost = ", ".join(item["title"] or item["doc_id"] for item in row["gold_lost"]) or "none"
        print(f"  {row['direction']:16s} {row['question_id']} {row['question_type']:28s} "
              f"docs_changed={row['documents_changed']}  gained={gained}  lost={lost}")
    print("\nQuestions whose candidate list gained a gold document")
    for row in body["gained_gold"]:
        kept = ", ".join(f"{item['title']} kept={item['kept']}" for item in row["gold"])
        print(f"  {row['question_id']} {row['question_type']}  before={row['correct_before']} "
              f"after={row['correct_after']}  top={row['phase23_top_score']:.2f}  {kept}")
    signal = body["runtime_signal"]
    print("\nPhase 2.3 top Jev score")
    print(f"{'group':28s}{'n':>6s}{'min':>8s}{'p25':>8s}{'median':>8s}{'p75':>8s}{'max':>8s}{'mean':>8s}")
    labels = {
        "a_gold_in_correct": "a gold in, correct",
        "b_gold_in_wrong": "b gold in, wrong",
        "c_gold_missing": "c gold missing",
    }
    for key, label in labels.items():
        dist = signal["distribution"][key]
        if not dist["n"]:
            print(f"{label:28s}{0:6d}")
            continue
        print(f"{label:28s}{dist['n']:6d}{dist['min']:8.2f}{dist['p25']:8.2f}{dist['median']:8.2f}"
              f"{dist['p75']:8.2f}{dist['max']:8.2f}{dist['mean']:8.2f}")
    print(f"DEV questions with no gold document: {len(signal['excluded_no_gold'])}")
    print(f"group c answers that were still correct: {signal['correct_with_gold_missing']}")
    print(f"{'top score below':20s}{'questions':>12s}{'of them group c':>18s}")
    for cutoff, row in signal["thresholds"].items():
        print(f"{cutoff:20s}{row['below']:12d}{row['group_c']:18d}")
    print("\nPhase 2.3 top score of questions expansion fixed")
    for row in signal["fixed_top_scores"]:
        print(f"  {row['question_id']} {row['question_type']} {row['phase23_top_score']:.2f}")


def _report(dev_ids, guard_ids, questions, prepared_base, prepared_new, probes_note) -> dict:
    titles = _titles()
    gold = {q.question_id: list(q.gold) for q in questions}
    old_docs = {qid: prepared_base[qid]["doc_ids"] for qid in dev_ids + guard_ids}
    new_docs = {qid: prepared_new[qid]["doc_ids"] for qid in dev_ids + guard_ids}
    before = _baseline_answers(dev_ids, guard_ids)
    after = _load_eval(OUT_DIR / "jev" / "answers_eval.jsonl")
    missing = [qid for qid in dev_ids + guard_ids if qid not in after]
    if missing:
        raise RuntimeError(f"New Jev rows missing for {missing[:5]}.")
    dev_flips = _flips(before, after, dev_ids, old_docs, new_docs, gold, titles)
    guard_flips = _flips(before, after, guard_ids, old_docs, new_docs, gold, titles)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"wrong_to_right": 0, "right_to_wrong": 0})
    for row in dev_flips:
        by_type[row["question_type"]][row["direction"]] += 1
    dev_q = [q for q in questions if q.question_id in set(dev_ids)]
    base_ret = _retrieval(dev_q, prepared_base)
    new_ret = _retrieval(dev_q, prepared_new)
    dev_before = aggregate([before[qid] for qid in dev_ids])
    dev_after = aggregate([after[qid] for qid in dev_ids])
    guard_before = aggregate([before[qid] for qid in guard_ids])
    guard_after = aggregate([after[qid] for qid in guard_ids])
    answer_cost = json.loads((OUT_DIR / "answer_cost.json").read_text())
    judge = json.loads((OUT_DIR / "jev" / "answers_metrics.json").read_text())
    jev_cost = json.loads((OUT_DIR / "jev_cost.json").read_text())
    body = {
        "baseline": "phase 2.3 + floor line",
        "probe": "hypothetical passage is search-only; excluded from the Jev state and the answer prompt",
        "costs": {
            "jev_usd": jev_cost.get("total_cost_usd"),
            "answer_usd": answer_cost.get("total_cost_usd"),
            "judge_usd": judge.get("judge_cost_usd"),
            "answer_seconds": answer_cost.get("seconds"),
            "answer_failures": answer_cost.get("failures"),
        },
        "expansion_per_question": probes_note,
        "retrieval": {"baseline": base_ret, "expanded": new_ret},
        "dev": {
            "baseline": dev_before["overall"],
            "expanded": dev_after["overall"],
            "baseline_by_type": dev_before["by_type"],
            "expanded_by_type": dev_after["by_type"],
            "wrong_to_right": sum(1 for row in dev_flips if row["direction"] == "wrong_to_right"),
            "right_to_wrong": sum(1 for row in dev_flips if row["direction"] == "right_to_wrong"),
            "flips_by_type": by_type,
        },
        "info_not_found": {
            "baseline": guard_before["overall"],
            "expanded": guard_after["overall"],
            "wrong_to_right": sum(1 for row in guard_flips if row["direction"] == "wrong_to_right"),
            "right_to_wrong": sum(1 for row in guard_flips if row["direction"] == "right_to_wrong"),
        },
        "flips": dev_flips + guard_flips,
        "gained_gold": _gained_gold(
            _saved_dev(), dev_q, new_docs, after, before, _top_scores(), titles),
        "runtime_signal": _runtime_signal(dev_ids, before, old_docs, gold, dev_flips),
    }
    REPORT_PATH.write_text(json.dumps(body, indent=2))
    _print_report(body)
    return body


async def _answer(prepared: dict[str, dict], probes: dict[str, str], wanted: set[str]) -> int:
    by_text = {item["question"]: qid for qid, item in prepared.items() if qid in wanted}
    if len(by_text) != len(wanted):
        raise RuntimeError("Two questions share the same text; answering keys on the text.")
    llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens)

    async def answer(question: str) -> dict:
        item = prepared[by_text[question]]
        texts = [p["text"] for p in item["passages"]]
        assert_probe_excluded(probes[by_text[question]], item["question"], texts, "the answer prompt")
        context = "\n\n".join(texts)
        if item["floor"]:
            pairs = [("system", SYSTEM_PROMPT), ("user", user_prompt(item["question"], context))]
            version = FLOOR_VERSION
        else:
            pairs = messages(item["question"], context)
            version = ANSWER_PROMPT_VERSION
        res = await llm.ainvoke(pairs, prompt_version=version)
        return {"answer": res.text, "doc_ids": item["doc_ids"]}

    usage.reset()
    args = argparse.Namespace(
        limit=None, question_type=None, question_types=None,
        dry_run=False, evaluate=True, concurrency=5, only_empty=False)
    code = await _run_phase("p2_expand_passage_dev", answer, args, OUT_DIR, "jev", wanted)
    print("\n=== answering cost ===")
    print((OUT_DIR / "answer_cost.json").read_text())
    metrics = OUT_DIR / "jev" / "answers_metrics.json"
    if metrics.is_file():
        print(f"judge cost ${json.loads(metrics.read_text()).get('judge_cost_usd')}")
    return code


async def main_async(dry_run: bool) -> int:
    dev_qs, guard_qs, dev_ids, guard_ids = _dev_and_guard()
    saved = _saved_dev()
    if set(saved) != set(dev_ids):
        raise RuntimeError("The DEV expansion file does not match the DEV split.")
    probes = {qid: row["hypothetical_passage"] for qid, row in saved.items()}
    saved_lists = {qid: row["lists"]["2"] for qid, row in saved.items()}
    guard_rows, guard_cost, guard_seconds = await _guard_probes(guard_qs, generate=not dry_run)
    if guard_rows is None:
        print("\nGuard passages are not on disk. Jev and answering dry-runs below cover DEV only.")
        guard_qs = []
        guard_rows = {}
    probes.update(guard_rows)
    questions = _build(dev_qs + guard_qs, probes, saved_lists)
    index = DocIndex()
    answer_bound = _answer_tokens(questions, probes, index)
    print("\n=== answering dry-run upper bound ===")
    print("Context is every passage that fits in the 10k budget, before the relevance cutoff. "
          "Output assumes the full 4,000-token cap. The floor line is included.")
    print(json.dumps(answer_bound, indent=2))

    usage.reset()
    await score_passages(questions, dry_run=True)
    jev_bound = usage.report()
    model = next(iter(jev_bound["models"].values()))
    local = jev_bound.get("total_estimated_cost_usd") or 0
    print("\n=== Jev scoring dry-run (cached calls are free) ===")
    print(json.dumps(jev_bound, indent=2))
    print(f"calibrated uncached cost ${round(local * BILLED_TOKEN_RATIO, 4)} "
          f"(local tokens x {BILLED_TOKEN_RATIO}); "
          f"{model.get('cached_calls', 0)} cache hits, {model.get('estimated_calls', 0)} new calls")
    if dry_run:
        return 0

    print("\n=== starting paid Jev scoring, then answering ===")
    usage.reset()
    probs = _load_scores(questions)
    if probs is None:
        probs = await score_passages(questions, dry_run=False)
        if any(p != p for values in probs.values() for p in values):  # NaN
            raise RuntimeError("Jev scoring left a missing probability.")
        _save_scores(questions, probs)
        jev_paid = usage.report()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "jev_cost.json").write_text(json.dumps(jev_paid, indent=2))
    else:
        logger.info("Loaded saved Jev scores.")
        jev_paid = json.loads((OUT_DIR / "jev_cost.json").read_text()) if (OUT_DIR / "jev_cost.json").is_file() else {}
    prepared = prepare(questions, probs, index)
    for q in questions:
        texts = [p["text"] for p in prepared[q.question_id]["passages"]]
        assert_probe_excluded(probes[q.question_id], q.question, texts, "the answer prompt")
    code = await _answer(prepared, probes, {q.question_id for q in questions})
    if code != 0:
        return code
    base_questions = [q for q in load_fused()[0] if q.question_id in set(dev_ids) | set(guard_ids)]
    prepared_base = prepare(base_questions, load_fused()[1], index)
    note = _expansion_cost(_dev_generation_cost(), guard_cost, guard_seconds, len(dev_ids), len(guard_ids))
    _report(dev_ids, guard_ids, questions, prepared_base, prepared, note)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer DEV + guard from the hypothetical-passage candidate list.")
    parser.add_argument("--dry-run", action="store_true", help="Price Jev scoring and answering, then stop.")
    args = parser.parse_args(argv)
    setup_logging()
    return asyncio.run(main_async(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
