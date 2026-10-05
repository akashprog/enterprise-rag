"""Step 2.3: offer the Phase 2.2 filter a different candidate list.

Passage shape, the Jev rule (p >= 0.7, at least one document, 10k tokens) and
the prompt stay as Phase 2.2. Part 1 only compares lists and calls no model.
Part 2 scores the picked list, fused top 30, and stops before answering.

Paid questions are two groups. The rest wait for the Phase 4 router:
    focus   basic, semantic, intra_document_reasoning, constrained,
            conflicting_info, miscellaneous, high_level (420 questions)
    guard   info_not_found (20). Scored, reported on its own, kept out of the
            headline so a refusal question cannot move it.
    parked  completeness, project_related. Part 1's free tables still include them.

Usage:
    python -m phase_2_data_mastery.hybrid_candidates                 # Part 1, $0
    python -m phase_2_data_mastery.hybrid_candidates --part2 --dry-run
    python -m phase_2_data_mastery.hybrid_candidates --part2         # score, then report
    python -m phase_2_data_mastery.hybrid_candidates --stability --dry-run
    python -m phase_2_data_mastery.hybrid_candidates --stability     # rescore 20 shuffled calls
    python -m phase_2_data_mastery.hybrid_candidates --part3         # answer focus + guard, Jev judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.ingest_naive import load_docs  # noqa: E402
from phase_2_data_mastery.hybrid_check import build_bm25, fuse  # noqa: E402
from phase_2_data_mastery.query_small_to_big import (  # noqa: E402
    DEFAULT_WHOLE_DOC_MAX,
    DEFAULT_WINDOW,
    DocIndex,
    messages,
    passage_for,
)
from phase_2_data_mastery.relevance_filter import (  # noqa: E402
    BILLED_TOKEN_RATIO,
    DEFAULT_BUDGET,
    RELEVANCE_PROMPT_VERSION,
    Candidate,
    QuestionCandidates,
    attach_passages,
    estimate_cost,
    load_candidates,
    load_scores,
    passage_text,
    prompt_tokens_from_counts,
    relevance_batches,
    score_passages,
    select_by_probability,
    summarise_kept,
    _CallPacer,
    _one_batch,
)
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import QuestionScore, aggregate, dedupe, format_table  # noqa: E402
from shared_utils.jev_judge import CachedJev  # noqa: E402
from shared_utils.llm import CachedChat, run_limited, truncate_to_tokens, usage  # noqa: E402
from shared_utils.runner import run_phase  # noqa: E402

logger = logging.getLogger("phase2.hybrid_candidates")

# The comparison point for every later number in this step.
PHASE_22_EVAL = settings.paths.results_dir / "phase_2" / "p07_f1_k30_b10000" / "jev" / "answers_eval.jsonl"
OUT_DIR = settings.paths.results_dir / "phase_2" / "hybrid_candidates"
# Same depth the earlier fusion check used. Vector search saved 50 documents.
FUSION_DEPTH = 50
PASS_AT = 0.7

FOCUS = ("basic", "semantic", "intra_document_reasoning", "constrained",
         "conflicting_info", "miscellaneous", "high_level")
GUARD = ("info_not_found",)
# What a paid Jev scoring run would cover. Parked types are counted in the
# free gold-found tables and left out of the cost.
PAID = FOCUS + GUARD
SETS = ("vector_top30", "fused_top30", "union_20", "union_30")


def phase22_on_focus() -> dict:
    """Phase 2.2's Jev scores, restricted to the focus set and the guard.

    Read from the saved evaluation. No judge is called.
    """
    rows = [json.loads(line) for line in PHASE_22_EVAL.open(encoding="utf-8")]
    focus = [QuestionScore(**row) for row in rows if row["question_type"] in FOCUS]
    guard = [QuestionScore(**row) for row in rows if row["question_type"] in GUARD]
    if len(focus) != 420 or len(guard) != 20:
        raise RuntimeError(f"Expected 420 focus and 20 guard questions, got {len(focus)} and {len(guard)}.")
    return {"focus": aggregate(focus), "guard": aggregate(guard)}


def own_distractors() -> dict[str, set[str]]:
    """question_id -> doc_ids added as hard negatives for that question."""
    out: dict[str, set[str]] = defaultdict(set)
    for row in load_docs(None).itertuples():
        qids = row.distractor_for if isinstance(row.distractor_for, list) else []
        for qid in qids:
            out[qid].add(row.doc_id)
    return out


def candidate_lists(q, bm25_rows: list[dict]) -> dict[str, list[str]]:
    """The four lists under comparison, each a doc_id sequence.

    Fusion is the same rule as the free hybrid check: reciprocal rank fusion
    of each method's top 50, then the first 30. A union keeps vector order and
    appends BM25 documents the vector list did not already contain.
    """
    vector = [c.doc_id for c in q.candidates]
    bm25 = [row["doc_id"] for row in bm25_rows]
    return {
        "vector_top30": vector[:30],
        "fused_top30": fuse(vector[:FUSION_DEPTH], bm25[:FUSION_DEPTH], limit=30),
        "union_20": dedupe(vector[:20] + bm25[:20]),
        "union_30": dedupe(vector[:30] + bm25[:30]),
    }


def gold_in(questions, lists: dict[str, dict[str, list[str]]], name: str, *, types: tuple[str, ...] | None) -> dict:
    """Gold documents present anywhere in the candidate list, not only the first 10."""
    found = asked = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_question: list[float] = []
    for q in questions:
        if types is not None and q.question_type not in types:
            continue
        gold = dedupe(q.gold)
        if not gold:
            continue
        hit = set(lists[q.question_id][name])
        n_hit = sum(doc_id in hit for doc_id in gold)
        found += n_hit
        asked += len(gold)
        per_question.append(n_hit / len(gold))
        by_type[q.question_type][0] += n_hit
        by_type[q.question_type][1] += len(gold)
    return {
        "found": found, "asked": asked,
        "pct": round(100 * found / asked, 1) if asked else None,
        "per_question_pct": round(100 * sum(per_question) / len(per_question), 1) if per_question else None,
        "by_type": {
            qtype: {"found": n[0], "asked": n[1], "pct": round(100 * n[0] / n[1], 1)}
            for qtype, n in sorted(by_type.items())
        },
    }


def list_sizes(questions, lists, name: str) -> dict[str, float]:
    counts = [len(lists[q.question_id][name]) for q in questions]
    return {"avg": round(sum(counts) / len(counts), 2), "max": max(counts)}


def passage_for_doc(doc_id: str, q, bm25_by_id: dict[str, dict], index: DocIndex) -> str:
    """The Phase 2.2 passage for one candidate.

    A document the vector search already returned keeps that search's chunks,
    so its text matches the call already in the Jev cache. A document only
    BM25 found uses the same whole-or-window rule around its own matched chunks.
    """
    vector = next((c for c in q.candidates if c.doc_id == doc_id), None)
    if vector is not None:
        return passage_text(vector, index)
    row = bm25_by_id[doc_id]
    _, text, _ = passage_for(index.get(row["path"]), row["chunks"],
                             window=DEFAULT_WINDOW, whole_doc_max=DEFAULT_WHOLE_DOC_MAX)
    return text


def new_candidate_cost(questions, lists: dict[str, dict[str, list[str]]],
                       bm25_full: dict[str, list[dict]]) -> dict:
    """Uncached Jev requests on the focus set and the guard, and what scoring them would cost.

    A request holds every passage of one question that fits in one call. The
    local token count is what `--dry-run` will print. The calibrated cost
    multiplies by the v1 billed/local ratio, because these v2 calls are not
    in the cache yet.
    """
    jev = CachedJev()
    index = DocIndex()
    price = settings.model_prices.get(settings.jev_model) or settings.model_prices.get("jev-latest")
    paid = [q for q in questions if q.question_type in PAID]
    out: dict[str, dict] = {}

    for name in SETS:
        passages = calls = uncached = local_new = 0
        for q in paid:
            bm25_by_id = {row["doc_id"]: row for row in bm25_full[q.question_id]}
            texts = [passage_for_doc(doc_id, q, bm25_by_id, index) for doc_id in lists[q.question_id][name]]
            passages += len(texts)
            for batch in relevance_batches(q.question, texts):
                calls += 1
                hit = jev.peek(batch["state"], batch["questions"], prompt_version=RELEVANCE_PROMPT_VERSION)
                if hit is None:
                    uncached += 1
                    local_new += batch["local_tokens"]
        estimate = None if price is None else round(local_new * price.input / 1e6, 4)
        calibrated = None if price is None else round(local_new * BILLED_TOKEN_RATIO * price.input / 1e6, 4)
        out[name] = {"candidates": passages, "calls": calls, "new": uncached, "local_tokens": local_new,
                     "dry_run_usd": estimate, "calibrated_usd": calibrated, "token_ratio": BILLED_TOKEN_RATIO}
    return out


def pack_group(probs: list[float]) -> dict | None:
    """Pass rate at 0.7 and the median probability. None when the group is empty."""
    if not probs:
        return None
    ordered = sorted(probs)
    passed = sum(p >= PASS_AT for p in ordered)
    return {"n": len(ordered), "pass_pct": round(100 * passed / len(ordered), 1),
            "median": round(statistics.median(ordered), 3)}


def jev_groups(questions, probs: dict[str, list[float]], distractors: dict[str, set[str]]) -> dict:
    """Phase 2.2's cached scores, split into gold, this question's distractors, and the rest."""
    overall: dict[str, list[float]] = {"gold": [], "own_distractor": [], "other": []}
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"gold": [], "own_distractor": [], "other": []})
    for q in questions:
        gold = set(dedupe(q.gold))
        own = distractors.get(q.question_id, set()) - gold
        for cand, p in zip(q.candidates[:30], probs[q.question_id]):
            if cand.doc_id in gold:
                group = "gold"
            elif cand.doc_id in own:
                group = "own_distractor"
            else:
                group = "other"
            overall[group].append(p)
            by_type[q.question_type][group].append(p)
    return {
        "overall": {name: pack_group(values) for name, values in overall.items()},
        "by_type": {
            qtype: {name: pack_group(values) for name, values in groups.items()}
            for qtype, groups in sorted(by_type.items())
        },
    }


# Part 2 sweeps these cutoffs. 0.7 is the Phase 2.2 setting; 0.95 is the extra one.
PART2_THRESHOLDS = (0.7, 0.8, 0.9, 0.95)
PART2_ORDERS = ("search", "jev")
FLOOR = 1
SCORES_PATH = OUT_DIR / "fused_top30_scores.jsonl"
# Distinct from Phase 2.2's "p2-relevance-v1", so a fused prompt cannot be
# served from that run's answer cache.
ANSWER_PROMPT_VERSION = "p2-hybrid-v1"
RUN_NAME = "fused30_p07_jev"
STABILITY_N = 20
STABILITY_SEED = 23
GROUPS = ("gold", "own_distractor", "other")


def fused_questions(questions, bm25_full: dict[str, list[dict]], index: DocIndex) -> list[QuestionCandidates]:
    """Focus set and guard, each with its fused top 30 as the candidate list.

    A document the vector search already returned keeps that candidate, so the
    passage is the same text Phase 2.2 would have built. A document only BM25
    found gets the same whole-or-window rule around its own matched chunks.
    """
    out: list[QuestionCandidates] = []
    for q in questions:
        if q.question_type not in PAID:
            continue
        vector_by_id = {c.doc_id: c for c in q.candidates}
        bm25_rows = bm25_full[q.question_id]
        bm25_by_id = {row["doc_id"]: row for row in bm25_rows}
        ids = fuse([c.doc_id for c in q.candidates][:FUSION_DEPTH],
                   [row["doc_id"] for row in bm25_rows][:FUSION_DEPTH], limit=30)
        cands: list[Candidate] = []
        fresh: list[Candidate] = []
        for doc_id in ids:
            if doc_id in vector_by_id:
                cands.append(vector_by_id[doc_id])
                continue
            row = bm25_by_id[doc_id]
            cand = Candidate(doc_id, row["path"], float(row["score"]), list(row["chunks"]), row["source_type"])
            cands.append(cand)
            fresh.append(cand)
        attach_passages(fresh, index)
        out.append(QuestionCandidates(q.question_id, q.question_type, q.question, list(q.gold), cands))
    return out


def vector_top30(questions) -> list[QuestionCandidates]:
    """Phase 2.2's pool: the first 30 vector candidates, for the same questions."""
    return [QuestionCandidates(q.question_id, q.question_type, q.question, list(q.gold), q.candidates[:30])
            for q in questions if q.question_type in PAID]


def _of_types(questions, types: tuple[str, ...]) -> list:
    return [q for q in questions if q.question_type in types]


def select_sent(q, probs: list[float], *, threshold: float, order: str):
    """Documents the filter keeps. `search` is fused order; `jev` is highest probability first."""
    cands = list(q.candidates)
    ps = list(probs)
    if len(ps) != len(cands):
        raise RuntimeError(f"{q.question_id} has {len(cands)} candidates and {len(ps)} scores.")
    if order == "jev":
        ranked = sorted(range(len(ps)), key=lambda i: -ps[i])
        cands = [cands[i] for i in ranked]
        ps = [ps[i] for i in ranked]
    elif order != "search":
        raise ValueError(order)
    return select_by_probability(cands, ps, threshold=threshold, budget=DEFAULT_BUDGET, floor=FLOOR)


def selection_stats(questions, probs: dict[str, list[float]], distractors: dict[str, set[str]], *,
                    threshold: float, order: str) -> dict:
    """Recall, invalid docs, documents kept, prompt size, and own-distractors kept."""
    kept: dict[str, list[str]] = {}
    token_total = 0
    own_counts: list[int] = []
    for q in questions:
        sent = select_sent(q, probs[q.question_id], threshold=threshold, order=order)
        kept[q.question_id] = [c.doc_id for c, _, _ in sent]
        token_total += prompt_tokens_from_counts(q.question, [n for _, _, n in sent])
        gold = set(dedupe(q.gold))
        own = distractors.get(q.question_id, set()) - gold
        own_counts.append(sum(c.doc_id in own for c, _, _ in sent))
    stats = summarise_kept(questions, kept)
    n = len(questions)
    stats["prompt_tokens_avg"] = round(token_total / n) if n else None
    stats["est_answer_cost_usd"] = estimate_cost(token_total, n) if n else None
    stats["own_distractor_avg"] = round(statistics.mean(own_counts), 3) if own_counts else None
    stats["kept_ids"] = kept
    return stats


def _group_of(doc_id: str, gold: set[str], own: set[str]) -> str:
    if doc_id in gold:
        return "gold"
    if doc_id in own:
        return "own_distractor"
    return "other"


def collect_groups(questions, probs: dict[str, list[float]], distractors: dict[str, set[str]]):
    """Probabilities split into gold, this question's distractors, and the rest."""
    overall: dict[str, list[float]] = {name: [] for name in GROUPS}
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: {name: [] for name in GROUPS})
    for q in questions:
        gold = set(dedupe(q.gold))
        own = distractors.get(q.question_id, set()) - gold
        for cand, p in zip(q.candidates, probs[q.question_id]):
            group = _group_of(cand.doc_id, gold, own)
            overall[group].append(p)
            by_type[q.question_type][group].append(p)
    return overall, by_type


def rate_block(values: list[float]) -> dict | None:
    """n, median, and the pass rate at each Part 2 threshold."""
    if not values:
        return None
    return {
        "n": len(values),
        "median": round(statistics.median(values), 3),
        "pass_pct": {
            f"{t:g}": round(100 * sum(p >= t for p in values) / len(values), 1)
            for t in PART2_THRESHOLDS
        },
    }


def gold_fate(questions, probs: dict[str, list[float]], kept_ids: dict[str, list[str]], *,
              threshold: float) -> dict[str, dict]:
    """Gold documents in the candidate list, and how many the filter does not keep.

    `below` scored under the threshold. `not_kept` is absent from the prompt,
    which also includes a document the budget dropped. The floor can keep a
    document that scored under the threshold when it is first in the order used.
    """
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"in_pool": 0, "below": 0, "not_kept": 0})
    for q in questions:
        gold = set(dedupe(q.gold))
        if not gold:
            continue
        kept = set(kept_ids[q.question_id])
        for cand, p in zip(q.candidates, probs[q.question_id]):
            if cand.doc_id not in gold:
                continue
            row = by_type[q.question_type]
            row["in_pool"] += 1
            if p < threshold:
                row["below"] += 1
            if cand.doc_id not in kept:
                row["not_kept"] += 1
    return dict(sorted(by_type.items()))


def kept_histogram(questions, probs, *, threshold: float, order: str) -> dict:
    """How many documents each question keeps, and how often each count occurs."""
    counts = [len(select_sent(q, probs[q.question_id], threshold=threshold, order=order)) for q in questions]
    hist = Counter(counts)
    return {
        "n": len(counts),
        "avg": round(statistics.mean(counts), 2) if counts else None,
        "median": statistics.median(counts) if counts else None,
        "max": max(counts) if counts else None,
        "histogram": {str(k): hist[k] for k in sorted(hist)},
    }


def _require_scores(questions, probs: dict[str, list[float]]) -> None:
    missing = [q.question_id for q in questions
               if any(isinstance(p, float) and math.isnan(p) for p in probs[q.question_id])]
    if missing:
        raise RuntimeError(f"Missing Jev scores for {len(missing)} questions, first {missing[:3]}.")


def save_fused_scores(questions, probs: dict[str, list[float]]) -> None:
    """Candidates and probabilities together, so the report can be rebuilt from this file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SCORES_PATH.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({
                "question_id": q.question_id,
                "question_type": q.question_type,
                "question": q.question,
                "gold": q.gold,
                "candidates": [c.to_json() for c in q.candidates],
                "probs": [round(p, 4) for p in probs[q.question_id]],
            }) + "\n")


def part2(*, dry_run: bool) -> int:
    """Score the fused top 30 on the focus set and the guard, then report. No answers."""
    questions = load_candidates()
    phase22_on_focus()
    index_bm25, meta, _ = build_bm25()
    paid = [q for q in questions if q.question_type in PAID]
    bm25_full = {q.question_id: index_bm25.ranked_documents(q.question, meta, limit=FUSION_DEPTH) for q in paid}
    fused = fused_questions(questions, bm25_full, DocIndex())
    if len(fused) != 440:
        raise RuntimeError(f"Expected 440 questions to score, got {len(fused)}.")

    usage.reset()
    logger.info("Scoring fused top 30 for %d questions with %s.", len(fused), RELEVANCE_PROMPT_VERSION)
    probs = asyncio.run(score_passages(fused, dry_run=dry_run))
    for line in usage.summary_lines():
        print(f"[cost] {line}")
    if dry_run:
        return 0
    _require_scores(fused, probs)
    save_fused_scores(fused, probs)
    logger.info("Wrote %s", SCORES_PATH)

    distractors = own_distractors()
    v1 = load_scores()
    phase22_pool = vector_top30(questions)
    for q in phase22_pool:
        if len(v1[q.question_id]) < len(q.candidates):
            raise RuntimeError(f"Phase 2.2 scores for {q.question_id} are shorter than its top 30.")
        v1[q.question_id] = v1[q.question_id][:len(q.candidates)]

    focus_fused = _of_types(fused, FOCUS)
    focus_v1 = _of_types(phase22_pool, FOCUS)
    guard_fused = _of_types(fused, GUARD)
    guard_v1 = _of_types(phase22_pool, GUARD)

    phase22 = selection_stats(focus_v1, v1, distractors, threshold=PASS_AT, order="search")
    sweep = []
    chosen = None
    for order in PART2_ORDERS:
        for threshold in PART2_THRESHOLDS:
            stats = selection_stats(focus_fused, probs, distractors, threshold=threshold, order=order)
            if order == "search" and threshold == PASS_AT:
                chosen = stats
            sweep.append(_sweep_row(stats, order=order, threshold=threshold))
    if chosen is None:
        raise RuntimeError("The Phase 2.2 setting is missing from the sweep.")

    overall_groups, by_type_groups = collect_groups(fused, probs, distractors)
    focus_groups, focus_by_type = collect_groups(focus_fused, probs, distractors)
    rates = {
        "focus_and_guard": {name: rate_block(values) for name, values in overall_groups.items()},
        "focus": {name: rate_block(values) for name, values in focus_groups.items()},
        "by_type": {
            qtype: {name: rate_block(values) for name, values in groups.items()}
            for qtype, groups in focus_by_type.items()
        },
    }
    report = {
        "candidate_set": "fused_top30",
        "prompt_version": RELEVANCE_PROMPT_VERSION,
        "questions": {"focus": len(focus_fused), "guard": len(guard_fused)},
        "cost": usage.report(),
        "phase22_focus": _sweep_row(phase22, order="search", threshold=PASS_AT),
        "sweep_focus": sweep,
        "gold_rejected": {
            "phase22": gold_fate(focus_v1, v1, phase22["kept_ids"], threshold=PASS_AT),
            "fused": gold_fate(focus_fused, probs, chosen["kept_ids"], threshold=PASS_AT),
        },
        "pass_rates": rates,
        "info_not_found": {
            "phase22": kept_histogram(guard_v1, v1, threshold=PASS_AT, order="search"),
            "fused": {
                f"{order}_{t:g}": kept_histogram(guard_fused, probs, threshold=t, order=order)
                for order in PART2_ORDERS for t in PART2_THRESHOLDS
            },
        },
        "by_type_focus": {
            "phase22": phase22["by_type"],
            "fused_search_0.7": chosen["by_type"],
        },
    }
    # kept_ids are large and already implied by the scores file.
    (OUT_DIR / "part2.json").write_text(json.dumps(report, indent=2))
    _print_part2(report)
    return 0


def _sweep_row(stats: dict, *, order: str, threshold: float) -> dict:
    overall = stats["overall"]
    return {
        "order": order,
        "threshold": threshold,
        "n": overall["n"],
        "recall10": overall["recall10"],
        "set_recall": overall["set_recall"],
        "invalid": overall["invalid"],
        "avg_docs": overall["avg_docs"],
        "docs_median": overall["docs_median"],
        "docs_p90": overall["docs_p90"],
        "own_distractor_avg": stats["own_distractor_avg"],
        "prompt_tokens_avg": stats["prompt_tokens_avg"],
        "est_answer_cost_usd": stats["est_answer_cost_usd"],
    }


def _print_part2(report: dict) -> None:
    print("\n=== Focus set, against the Phase 2.2 filter ===")
    print("Phase 2.2 is the vector top 30, scored one passage at a time. "
          "Fused rows are the RRF top 30, scored together.")
    print(f"{'setting':22s}{'R@10':>8s}{'set':>8s}{'invalid':>9s}{'avg':>7s}{'med':>6s}"
          f"{'p90':>6s}{'own':>7s}{'prompt':>8s}{'est$':>8s}")
    _setting_line("phase 2.2  p0.7", report["phase22_focus"])
    for row in report["sweep_focus"]:
        _setting_line(f"{row['order']:6s}  p{row['threshold']:g}", row)

    print("\nBy question type at p >= 0.7, search order (Phase 2.2 | fused):")
    print(f"{'type':28s}{'R@10':>16s}{'invalid':>16s}{'avg docs':>16s}")
    types = [t for t in FOCUS if t in report["by_type_focus"]["fused_search_0.7"]]
    for qtype in types:
        old, new = report["by_type_focus"]["phase22"][qtype], report["by_type_focus"]["fused_search_0.7"][qtype]
        print(f"{qtype:28s}{_pair(old.get('recall10'), new.get('recall10'))}"
              f"{_pair(old.get('invalid'), new.get('invalid'))}{_pair(old.get('avg_docs'), new.get('avg_docs'))}")

    print("\nGold documents in the pool at p >= 0.7, search order:")
    print(f"{'type':28s}{'p22 in pool':>12s}{'p22 below':>11s}{'p22 dropped':>12s}"
          f"{'fused in':>10s}{'fused below':>13s}{'fused dropped':>15s}")
    gold_types = list(report["gold_rejected"]["fused"])
    for qtype in gold_types:
        old = report["gold_rejected"]["phase22"].get(qtype, {"in_pool": 0, "below": 0, "not_kept": 0})
        new = report["gold_rejected"]["fused"][qtype]
        print(f"{qtype:28s}{old['in_pool']:12d}{old['below']:11d}{old['not_kept']:12d}"
              f"{new['in_pool']:10d}{new['below']:13d}{new['not_kept']:15d}")

    print("\nPass rates on the fused list (focus set + guard):")
    _print_rates(report["pass_rates"]["focus_and_guard"])
    print("Focus set only:")
    _print_rates(report["pass_rates"]["focus"])
    print("\nBy focus question type (pass% at 0.7 / 0.8 / 0.9 / 0.95):")
    for qtype in FOCUS:
        groups = report["pass_rates"]["by_type"].get(qtype)
        if groups is None:
            continue
        print(f"  {qtype}")
        for name in GROUPS:
            print(f"    {name:16s}{_rate_cell(groups[name])}")

    print("\ninfo_not_found, documents kept (not in the headline):")
    old = report["info_not_found"]["phase22"]
    print(f"  phase 2.2 p0.7 search: avg {old['avg']}, median {old['median']}, max {old['max']}, {old['histogram']}")
    for key, row in report["info_not_found"]["fused"].items():
        print(f"  fused {key:16s} avg {row['avg']}, median {row['median']}, max {row['max']}, {row['histogram']}")


def _setting_line(label: str, row: dict) -> None:
    print(f"{label:22s}{_num(row['recall10']):>8s}{_num(row['set_recall']):>8s}{_num(row['invalid']):>9s}"
          f"{_num(row['avg_docs']):>7s}{_num(row['docs_median']):>6s}{_num(row['docs_p90']):>6s}"
          f"{_num(row['own_distractor_avg']):>7s}{row['prompt_tokens_avg']:8d}{row['est_answer_cost_usd']:8.4f}")


def _num(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def _pair(old, new) -> str:
    left = "-" if old is None else f"{old:.1f}"
    right = "-" if new is None else f"{new:.1f}"
    return f"{left} | {right}".rjust(16)


def _print_rates(block: dict) -> None:
    print(f"{'group':18s}{'n':>8s}{'median':>8s}" + "".join(f"{t:>8g}" for t in PART2_THRESHOLDS))
    for name in GROUPS:
        row = block[name]
        if row is None:
            print(f"{name:18s}{'—':>8s}")
            continue
        cells = "".join(f"{row['pass_pct'][f'{t:g}']:7.1f}%" for t in PART2_THRESHOLDS)
        print(f"{name:18s}{row['n']:8d}{row['median']:8.3f}{cells}")


def _rate_cell(row: dict | None) -> str:
    if row is None:
        return "  —"
    cells = " / ".join(f"{row['pass_pct'][f'{t:g}']:.0f}%" for t in PART2_THRESHOLDS)
    return f"  n={row['n']:<5d} med {row['median']:.2f}   {cells}"


def load_fused() -> tuple[list[QuestionCandidates], dict[str, list[float]]]:
    """The fused top 30 and the probabilities Part 2 saved. No model calls."""
    if not SCORES_PATH.is_file():
        raise FileNotFoundError(f"{SCORES_PATH} not found. Run --part2 first.")
    questions: list[QuestionCandidates] = []
    probs: dict[str, list[float]] = {}
    for line in SCORES_PATH.open(encoding="utf-8"):
        raw = json.loads(line)
        cands = [Candidate(**{k: c[k] for k in (
            "doc_id", "path", "score", "chunks", "source_type", "passage_tokens", "mode")})
                 for c in raw["candidates"]]
        questions.append(QuestionCandidates(
            raw["question_id"], raw["question_type"], raw["question"], raw["gold"], cands))
        probs[raw["question_id"]] = raw["probs"]
    return questions, probs


def _batch_slices(q, index: DocIndex) -> list[tuple[int, list[str]]]:
    """The passages each original Jev call contained, in that call's order."""
    texts = [passage_text(c, index) for c in q.candidates]
    offset = 0
    slices: list[tuple[int, list[str]]] = []
    for batch in relevance_batches(q.question, texts):
        slices.append((offset, texts[offset:offset + batch["n"]]))
        offset += batch["n"]
    if offset != len(texts):
        raise RuntimeError(f"{q.question_id}: batches cover {offset} of {len(texts)} passages.")
    return slices


def _original_probs(q, index: DocIndex, jev: CachedJev) -> list[float]:
    """Full-precision probabilities from the Part 2 cache, in candidate order."""
    texts = [passage_text(c, index) for c in q.candidates]
    out: list[float | None] = [None] * len(texts)
    offset = 0
    for batch in relevance_batches(q.question, texts):
        hit = jev.peek(batch["state"], batch["questions"], prompt_version=RELEVANCE_PROMPT_VERSION)
        if hit is None:
            raise RuntimeError(f"Part 2 cache miss for {q.question_id}.")
        for local in range(batch["n"]):
            out[offset + local] = hit["answers"][f"p{local}"]["noul"]
        offset += batch["n"]
    return [float(p) for p in out]


async def _score_calls(batches: list[dict], *, dry_run: bool) -> list[dict | None]:
    """Send prepared Jev requests. Cached calls and a dry run do not wait."""
    jev = CachedJev(dry_run=dry_run)
    pacer = _CallPacer()

    async def one(batch: dict) -> dict | None:
        state, spec = batch["state"], batch["questions"]
        if not dry_run and jev.peek(state, spec, prompt_version=RELEVANCE_PROMPT_VERSION) is None:
            await pacer.wait(batch["local_tokens"])
        for attempt in range(4):
            try:
                return await jev.ask(state, spec, prompt_version=RELEVANCE_PROMPT_VERSION)
            except Exception as exc:  # noqa: BLE001 - transient rate limits and timeouts
                if attempt == 3 or dry_run:
                    raise
                logger.warning("Jev call failed (%s), retry %d", exc, attempt + 1)
                await asyncio.sleep(2 ** attempt)
        return None

    try:
        return await run_limited(batches, one, limit=8, desc="stability")
    finally:
        await jev.aclose()


def stability(*, dry_run: bool) -> int:
    """Rescore 20 focus questions with the passages shuffled inside each call.

    The set of passages in a call stays the same. Only their order changes, so
    a different probability is the effect of order, not of different neighbours.
    """
    questions, _ = load_fused()
    focus = sorted((q for q in questions if q.question_type in FOCUS), key=lambda q: q.question_id)
    picked = random.Random(STABILITY_SEED).sample(focus, STABILITY_N)
    picked.sort(key=lambda q: q.question_id)
    index = DocIndex()
    rng = random.Random(STABILITY_SEED)
    jobs: list[dict] = []
    for q in picked:
        for offset, texts in _batch_slices(q, index):
            perm = list(range(len(texts)))
            rng.shuffle(perm)
            jobs.append({
                "qid": q.question_id,
                "offset": offset,
                "perm": perm,
                "batch": _one_batch(q.question, [texts[i] for i in perm]),
            })

    usage.reset()
    logger.info("Stability: %d questions, %d calls, seed %d.", len(picked), len(jobs), STABILITY_SEED)
    results = asyncio.run(_score_calls([j["batch"] for j in jobs], dry_run=dry_run))
    for line in usage.summary_lines():
        print(f"[cost] {line}")
    if dry_run:
        return 0

    fresh: dict[str, list[float | None]] = {q.question_id: [None] * len(q.candidates) for q in picked}
    for job, result in zip(jobs, results):
        if result is None:
            raise RuntimeError(f"No score for a shuffled call on {job['qid']}.")
        for local, src in enumerate(job["perm"]):
            fresh[job["qid"]][job["offset"] + src] = result["answers"][f"p{local}"]["noul"]

    jev = CachedJev()
    distractors = own_distractors()
    counts = {name: {"n": 0, "crossed": 0} for name in GROUPS}
    largest: dict | None = None
    for q in picked:
        old = _original_probs(q, index, jev)
        gold = set(dedupe(q.gold))
        own = distractors.get(q.question_id, set()) - gold
        for cand, before, after in zip(q.candidates, old, fresh[q.question_id]):
            if after is None:
                raise RuntimeError(f"Missing shuffled score for {q.question_id} {cand.doc_id}.")
            group = _group_of(cand.doc_id, gold, own)
            counts[group]["n"] += 1
            if (before >= PASS_AT) != (after >= PASS_AT):
                counts[group]["crossed"] += 1
            delta = abs(after - before)
            if largest is None or delta > largest["abs"]:
                largest = {"abs": round(delta, 4), "before": round(before, 4), "after": round(after, 4),
                           "group": group, "question_id": q.question_id, "question_type": q.question_type,
                           "doc_id": cand.doc_id}
    report = {
        "seed": STABILITY_SEED,
        "questions": [{"question_id": q.question_id, "question_type": q.question_type} for q in picked],
        "calls": len(jobs),
        "cost": usage.report(),
        "crossed_0_7": counts,
        "largest_change": largest,
    }
    (OUT_DIR / "stability.json").write_text(json.dumps(report, indent=2))
    _print_stability(report)
    return 0


def _print_stability(report: dict) -> None:
    print("\n=== Stability: passage order shuffled inside each call ===")
    types = Counter(q["question_type"] for q in report["questions"])
    print(f"{report['calls']} calls over {len(report['questions'])} focus questions "
          f"({', '.join(f'{t} {n}' for t, n in sorted(types.items()))})")
    print(f"{'group':18s}{'passages':>10s}{'crossed 0.7':>14s}")
    for name in GROUPS:
        row = report["crossed_0_7"][name]
        print(f"{name:18s}{row['n']:10d}{row['crossed']:14d}")
    big = report["largest_change"]
    print(f"Largest |change|: {big['abs']:.4f} ({big['before']:.4f} -> {big['after']:.4f}), "
          f"{big['group']}, {big['question_id']} ({big['question_type']})")


def _sent_passages(q, probs: list[float], index: DocIndex) -> tuple[list[str], list[str]]:
    """Texts and doc ids the chosen rule sends, highest Jev score first."""
    texts: list[str] = []
    ids: list[str] = []
    for cand, mode, n_tokens in select_sent(q, probs, threshold=PASS_AT, order="jev"):
        text = passage_text(cand, index)
        if mode.endswith("+cut"):
            text = truncate_to_tokens(text, n_tokens)
        texts.append(text)
        ids.append(cand.doc_id)
    return texts, ids


def part3(*, only_empty: bool) -> int:
    """Answer the focus set and the guard, then judge with Jev. No cascade."""
    questions, probs = load_fused()
    focus_n = sum(q.question_type in FOCUS for q in questions)
    guard_n = sum(q.question_type in GUARD for q in questions)
    if focus_n != 420 or guard_n != 20 or len(questions) != 440:
        raise RuntimeError(f"Expected 420 focus and 20 guard, got {focus_n} and {guard_n}.")
    index = DocIndex()
    prepared = {q.question: _sent_passages(q, probs[q.question_id], index) for q in questions}
    if len(prepared) != len(questions):
        raise RuntimeError("Two questions share the same text; answering keys on the text.")
    llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens)

    async def answer(question: str) -> dict:
        texts, doc_ids = prepared[question]
        res = await llm.ainvoke(messages(question, "\n\n".join(texts)), prompt_version=ANSWER_PROMPT_VERSION)
        return {"answer": res.text, "doc_ids": doc_ids}

    out_dir = settings.paths.results_dir / "phase_2" / RUN_NAME
    args = argparse.Namespace(
        limit=None, question_type=None, question_types=",".join(PAID),
        dry_run=False, evaluate=True, concurrency=5, only_empty=only_empty)
    usage.reset()
    code = run_phase(RUN_NAME, answer, args, out_dir=out_dir, judge="jev")
    if code not in (0, 1):
        raise RuntimeError(f"Answering exited {code}")
    report = _part3_report(out_dir)
    (OUT_DIR / "part3.json").write_text(json.dumps(report, indent=2))
    _print_part3(report)
    return 0


def _part3_report(run_dir: Path) -> dict:
    """Focus-set comparison with Phase 2.2. The guard is kept on its own."""
    rows = [QuestionScore(**raw) for raw in map(json.loads, (run_dir / "jev" / "answers_eval.jsonl").open())]
    focus = [r for r in rows if r.question_type in FOCUS]
    guard = [r for r in rows if r.question_type in GUARD]
    if len(focus) != 420 or len(guard) != 20:
        raise RuntimeError(f"Expected 420 focus and 20 guard answers, got {len(focus)} and {len(guard)}.")
    baseline = phase22_on_focus()
    old = {raw["question_id"]: raw for raw in map(json.loads, PHASE_22_EVAL.open())}
    flips: dict[str, Counter] = defaultdict(Counter)
    for row in focus:
        before, after = bool(old[row.question_id]["correct"]), bool(row.correct)
        if not before and after:
            kind = "wrong_to_right"
        elif before and not after:
            kind = "right_to_wrong"
        elif after:
            kind = "both_right"
        else:
            kind = "both_wrong"
        flips[row.question_type][kind] += 1
        flips["FOCUS"][kind] += 1
    guard_flips: Counter = Counter()
    for row in guard:
        before, after = bool(old[row.question_id]["correct"]), bool(row.correct)
        if not before and after:
            guard_flips["wrong_to_right"] += 1
        elif before and not after:
            guard_flips["right_to_wrong"] += 1
        elif after:
            guard_flips["both_right"] += 1
        else:
            guard_flips["both_wrong"] += 1
    answer_cost = json.loads((run_dir / "answer_cost.json").read_text())
    judge = json.loads((run_dir / "jev" / "answers_metrics.json").read_text())
    filter_cost = json.loads((OUT_DIR / "part2.json").read_text())["cost"]["total_cost_usd"]
    return {
        "run": str(run_dir),
        "order": "jev",
        "threshold": PASS_AT,
        "focus": aggregate(focus),
        "phase22_focus": baseline["focus"],
        "flips_focus": {k: dict(v) for k, v in flips.items()},
        "guard": aggregate(guard),
        "phase22_guard": baseline["guard"],
        "flips_guard": dict(guard_flips),
        "answer_cost_usd": answer_cost.get("total_cost_usd"),
        "answer_seconds": answer_cost.get("seconds"),
        "answer_failures": answer_cost.get("failures"),
        "judge_cost_usd": judge.get("judge_cost_usd"),
        "filter_cost_usd": filter_cost,
    }


def _print_part3(report: dict) -> None:
    here, old = report["focus"]["overall"], report["phase22_focus"]["overall"]
    print("\n=== Focus set, answered. p>=0.7, floor 1, 10k, Jev-score order. Judge: Jev ===")
    print(f"{'':28s}{'overall':>10s}{'correct':>10s}{'complete':>10s}{'recall':>10s}{'invalid':>10s}{'words':>8s}")
    print(f"{'phase 2.2':28s}{_m(old, 'overall_score')}{_m(old, 'correctness')}{_m(old, 'completeness')}"
          f"{_m(old, 'recall_at_k')}{_m(old, 'invalid_extra_docs')}{_m(old, 'avg_answer_words', 8, '.1f')}")
    print(f"{'fused, Jev order':28s}{_m(here, 'overall_score')}{_m(here, 'correctness')}{_m(here, 'completeness')}"
          f"{_m(here, 'recall_at_k')}{_m(here, 'invalid_extra_docs')}{_m(here, 'avg_answer_words', 8, '.1f')}")
    flip = report["flips_focus"]["FOCUS"]
    print(f"flips vs phase 2.2: {flip.get('wrong_to_right', 0)} wrong->right, "
          f"{flip.get('right_to_wrong', 0)} right->wrong, "
          f"{flip.get('both_right', 0)} both right, {flip.get('both_wrong', 0)} both wrong")
    print(f"answer ${report['answer_cost_usd']} in {report['answer_seconds']}s, "
          f"{report['answer_failures']} failures")
    print(f"judge ${report['judge_cost_usd']}   filter ${report['filter_cost_usd']:.4f}")

    print(f"\n{'type':28s}{'p22':>8s}{'fused':>8s}{'recall':>8s}{'invalid':>8s}{'w->r':>6s}{'r->w':>6s}")
    for qtype in FOCUS:
        new = report["focus"]["by_type"][qtype]
        base = report["phase22_focus"]["by_type"][qtype]
        f = report["flips_focus"].get(qtype, {})
        print(f"{qtype:28s}{base['overall_score']:8.2f}{new['overall_score']:8.2f}"
              f"{_cell(new.get('recall_at_k'))}{_cell(new.get('invalid_extra_docs'))}"
              f"{f.get('wrong_to_right', 0):6d}{f.get('right_to_wrong', 0):6d}")

    g, og = report["guard"]["overall"], report["phase22_guard"]["overall"]
    gf = report["flips_guard"]
    print(f"\ninfo_not_found, separate: {g['overall_score']:.2f} overall, {g['correctness']:.2f} correct, "
          f"{g['n']} questions (phase 2.2 {og['overall_score']:.2f}). "
          f"flips {gf.get('wrong_to_right', 0)} wrong->right, {gf.get('right_to_wrong', 0)} right->wrong")


def _m(row: dict, key: str, width: int = 10, spec: str = ".2f") -> str:
    value = row.get(key)
    if value is None:
        return f"{'-':>{width}s}"
    return f"{value:{width}{spec}}"


def _cell(value) -> str:
    return f"{'-':>8s}" if value is None else f"{value:8.2f}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2.3 candidate lists. Part 2 scores the fused top 30.")
    p.add_argument("--part2", action="store_true", help="Score the fused top 30 on the focus set and the guard.")
    p.add_argument("--part3", action="store_true",
                   help="Answer the focus set and the guard at p>=0.7, Jev-score order, and judge with Jev.")
    p.add_argument("--stability", action="store_true",
                   help="Rescore 20 focus questions with passage order shuffled inside each call.")
    p.add_argument("--only-empty", action="store_true",
                   help="With --part3, re-answer only empty or failed answers.")
    p.add_argument("--dry-run", action="store_true",
                   help="With --part2 or --stability, price the Jev calls and do not make them.")
    args = p.parse_args(argv)
    if args.dry_run and not (args.part2 or args.stability):
        p.error("--dry-run prices --part2 or --stability")
    if args.only_empty and not args.part3:
        p.error("--only-empty is only for --part3")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    if args.stability:
        code = stability(dry_run=args.dry_run)
        if code != 0 or args.dry_run or not args.part3:
            return code
    if args.part3:
        return part3(only_empty=args.only_empty)
    if args.part2:
        return part2(dry_run=args.dry_run)
    baseline = phase22_on_focus()
    questions = load_candidates()
    probs = load_scores()
    groups = jev_groups(questions, probs, own_distractors())

    index, meta, _ = build_bm25()
    bm25_full = {q.question_id: index.ranked_documents(q.question, meta, limit=FUSION_DEPTH) for q in questions}
    lists = {}
    for q in questions:
        lists[q.question_id] = candidate_lists(q, bm25_full[q.question_id])

    gold = {
        name: {"all": gold_in(questions, lists, name, types=None),
               "focus": gold_in(questions, lists, name, types=FOCUS)}
        for name in SETS
    }
    sizes = {name: list_sizes(questions, lists, name) for name in SETS}
    cost = new_candidate_cost(questions, lists, bm25_full)

    report = {"phase22_focus": baseline, "gold_found": gold, "sizes": sizes,
              "new_candidates_focus_and_guard": cost, "jev_groups": groups}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "part1.json").write_text(json.dumps(report, indent=2))
    _print(report)
    return 0


def _print(report: dict) -> None:
    focus, guard = report["phase22_focus"]["focus"], report["phase22_focus"]["guard"]
    print("\n=== Phase 2.2 on the focus set (420 questions, from saved Jev verdicts) ===")
    print(format_table(focus))
    g = guard["overall"]
    print(f"info_not_found, separate: {g['overall_score']:.2f} overall, {g['correctness']:.2f} correct, "
          f"{g['n']} questions")

    print("\n=== A. Gold found in the candidate list (all types / focus set) ===")
    print(f"{'set':16s}{'all found':>16s}{'all / question':>16s}{'focus found':>16s}{'avg docs':>10s}{'max':>6s}")
    for name in SETS:
        all_g, focus_g = report["gold_found"][name]["all"], report["gold_found"][name]["focus"]
        size = report["sizes"][name]
        print(f"{name:16s}{_found(all_g):>16s}{all_g['per_question_pct']:15.1f}%"
              f"{_found(focus_g):>16s}{size['avg']:10.1f}{size['max']:6d}")

    print("\nGold found by question type (percent of gold documents):")
    types = list(report["gold_found"]["vector_top30"]["all"]["by_type"])
    print(f"{'type':28s}" + "".join(f"{name:>16s}" for name in SETS))
    for qtype in types:
        cells = []
        for name in SETS:
            row = report["gold_found"][name]["all"]["by_type"][qtype]
            cells.append(f"{row['pct']:15.1f}%")
        print(f"{qtype:28s}" + "".join(cells))

    print("\nNew Jev calls on the focus set + guard (440 questions), one request per question:")
    print(f"{'set':16s}{'passages':>10s}{'calls':>8s}{'uncached':>10s}{'dry-run $':>12s}{'calibrated $':>14s}")
    for name in SETS:
        row = report["new_candidates_focus_and_guard"][name]
        print(f"{name:16s}{row['candidates']:10d}{row['calls']:8d}{row['new']:10d}{row['dry_run_usd']:12.4f}"
              f"{row['calibrated_usd']:14.4f}")
    ratio = report["new_candidates_focus_and_guard"]["vector_top30"]["token_ratio"]
    print(f"Calibrated applies the v1 billed/local token ratio ({ratio}).")

    print("\n=== B. Phase 2.2 Jev scores at p >= 0.7 (cached, all types) ===")
    print(f"{'group':18s}{'n':>8s}{'pass':>8s}{'median':>10s}")
    for name, row in report["jev_groups"]["overall"].items():
        _group_line(name, row)
    print("\nBy question type (pass% / median), gold | own distractor | other:")
    for qtype, groups in report["jev_groups"]["by_type"].items():
        print(f"  {qtype:28s}{_group_cell(groups['gold'])}{_group_cell(groups['own_distractor'])}"
              f"{_group_cell(groups['other'])}")


def _found(block: dict) -> str:
    return f"{block['found']}/{block['asked']} ({block['pct']:.1f}%)"


def _group_line(name: str, row: dict | None) -> None:
    if row is None:
        print(f"{name:18s}{'—':>8s}")
        return
    print(f"{name:18s}{row['n']:8d}{row['pass_pct']:7.1f}%{row['median']:10.3f}")


def _group_cell(row: dict | None) -> str:
    if row is None:
        return f"{'—':>22s}"
    return f"{row['pass_pct']:6.1f}% / {row['median']:.2f}".rjust(22)


if __name__ == "__main__":
    raise SystemExit(main())
