"""Baseline for the 60 parked questions. Phase 2 final, not query expansion.

Phase 2 final is the Phase 2.3 list and filter, plus the floor line:

    fused top 30 → batched Jev filter (p >= 0.7, floor 1, Jev order, 10k)
    → passages → Phase 1 prompt with the floor line

The 60 questions are completeness and project_related. The headline judge is
cascade (Jev first, uncertain verdicts re-asked to sol). A Jev-only score of
the same answers is saved beside it.

Usage:
    python -m phase_2_data_mastery.parked_baseline --dry-run
    python -m phase_2_data_mastery.parked_baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.query_naive import SYSTEM_PROMPT  # noqa: E402
from phase_2_data_mastery.answer_prompt_v2 import parse_jev, prepare  # noqa: E402
from phase_2_data_mastery.floor_line import PROMPT_VERSION as FLOOR_VERSION, user_prompt  # noqa: E402
from phase_2_data_mastery.hybrid_candidates import (  # noqa: E402
    ANSWER_PROMPT_VERSION,
    FLOOR,
    FUSION_DEPTH,
    PASS_AT,
    select_sent,
)
from phase_2_data_mastery.hybrid_check import build_bm25, fuse  # noqa: E402
from phase_2_data_mastery.query_small_to_big import DocIndex, messages  # noqa: E402
from phase_2_data_mastery.relevance_filter import (  # noqa: E402
    BILLED_TOKEN_RATIO,
    DEFAULT_BUDGET,
    QuestionCandidates,
    attach_passages,
    load_candidates,
    passage_text,
    score_passages,
    select_by_probability,
    truncate_to_tokens,
)
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import QuestionScore, aggregate, evaluate_file, load_answers, load_qa, parse_args  # noqa: E402
from shared_utils.llm import CachedChat, count_tokens, usage  # noqa: E402
from shared_utils.runner import _run_phase  # noqa: E402

logger = logging.getLogger("phase2.parked")

OUT_DIR = settings.paths.results_dir / "phase_2" / "p2_parked_baseline"
SCORES_PATH = OUT_DIR / "scores.jsonl"
REPORT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "parked_baseline.json"
PARKED = ("completeness", "project_related")
_FIELDS = {f.name for f in fields(QuestionScore)}

# Work groups from the read of these 60 questions. Multi-part is the four
# project_related jobs plus "spell out the full process". Count/list is the
# remaining completeness questions.
_FAILURE = (
    "qst_0341", "qst_0343", "qst_0349", "qst_0350", "qst_0352",
    "qst_0363", "qst_0368", "qst_0369", "qst_0370", "qst_0376",
)
_RULE = (
    "qst_0342", "qst_0345", "qst_0346", "qst_0348", "qst_0353", "qst_0355", "qst_0356",
    "qst_0360", "qst_0361", "qst_0364", "qst_0371", "qst_0374", "qst_0378", "qst_0380",
)
_OPS = (
    "qst_0344", "qst_0347", "qst_0351", "qst_0354", "qst_0357", "qst_0358",
    "qst_0365", "qst_0366", "qst_0367", "qst_0373", "qst_0375", "qst_0377", "qst_0379",
)
_RECONCILE = ("qst_0359", "qst_0362", "qst_0372")
_PROCESS = (
    "qst_0431", "qst_0433", "qst_0440", "qst_0441", "qst_0442",
    "qst_0444", "qst_0445", "qst_0447",
)
_COUNT = (
    "qst_0432", "qst_0434", "qst_0435", "qst_0436", "qst_0437", "qst_0438",
    "qst_0439", "qst_0443", "qst_0446", "qst_0448", "qst_0449", "qst_0450",
)
WORK = {qid: "multi-part" for qid in _FAILURE + _RULE + _OPS + _RECONCILE + _PROCESS}
WORK.update({qid: "count/list" for qid in _COUNT})


def _parked() -> list:
    questions = [q for q in load_candidates() if q.question_type in PARKED]
    ids = [q.question_id for q in questions]
    if len(questions) != 60 or len(set(ids)) != 60 or set(ids) != set(WORK):
        raise RuntimeError(f"Expected the 60 tagged parked questions, got {len(questions)}.")
    return questions


def _fuse(questions) -> list:
    """Fused top 30, the same rule as Phase 2.3, for these questions only."""
    logger.info("Building fused top 30 for %d parked questions.", len(questions))
    bm25_index, meta, n_chunks = build_bm25()
    logger.info("BM25 index over %d chunks.", n_chunks)
    index = DocIndex()
    out = []
    for q in questions:
        vector = list(q.candidates[:FUSION_DEPTH])
        bm25_rows = bm25_index.ranked_documents(q.question, meta, limit=FUSION_DEPTH)
        ids = fuse([c.doc_id for c in vector], [row["doc_id"] for row in bm25_rows], limit=30)
        by_vector = {c.doc_id: c for c in vector}
        by_bm25 = {row["doc_id"]: row for row in bm25_rows}
        cands = []
        fresh = []
        for doc_id in ids:
            if doc_id in by_vector:
                cands.append(by_vector[doc_id])
                continue
            row = by_bm25[doc_id]
            cand = type(vector[0])(doc_id, row["path"], float(row["score"]), list(row["chunks"]), row["source_type"])
            cands.append(cand)
            fresh.append(cand)
        attach_passages(fresh, index)
        out.append(QuestionCandidates(q.question_id, q.question_type, q.question, list(q.gold), cands))
    return out


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


def _answer_bound(questions, index: DocIndex) -> dict:
    """Upper bound: every passage that fits in 10k, floor line included, full output cap."""
    in_tok = 0
    for q in questions:
        dummy = [1.0] * len(q.candidates)
        sent = select_by_probability(q.candidates, dummy, threshold=0.0, budget=DEFAULT_BUDGET, floor=1)
        texts = []
        for cand, mode, n_tokens in sent:
            text = passage_text(cand, index)
            if mode.endswith("+cut"):
                text = truncate_to_tokens(text, n_tokens)
            texts.append(text)
        pairs = [("system", SYSTEM_PROMPT), ("user", user_prompt(q.question, "\n\n".join(texts)))]
        in_tok += sum(count_tokens(text) for _, text in pairs)
    out_tok = len(questions) * settings.answer_max_tokens
    return {
        "questions": len(questions),
        "input_tokens": in_tok,
        "output_tokens_cap": out_tok,
        "usd": usage.price(settings.answer_model, in_tok, out_tok),
    }


def _eligible(q, probs: list[float]) -> list[str]:
    """Documents the probability rule keeps, before the token budget."""
    cands = list(q.candidates)
    ps = list(probs)
    order = sorted(range(len(ps)), key=lambda i: -ps[i])
    cands = [cands[i] for i in order]
    ps = [ps[i] for i in order]
    chosen = [i for i, p in enumerate(ps) if p >= PASS_AT]
    if len(chosen) < FLOOR:
        for i in range(len(cands)):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= FLOOR:
                break
    chosen.sort()
    return [cands[i].doc_id for i in chosen]


def _prompt_tokens(item: dict) -> int:
    context = "\n\n".join(p["text"] for p in item["passages"])
    if item["floor"]:
        pairs = [("system", SYSTEM_PROMPT), ("user", user_prompt(item["question"], context))]
    else:
        pairs = messages(item["question"], context)
    return sum(count_tokens(text) for _, text in pairs)


def _layout(questions, probs, prepared: dict[str, dict]) -> dict[str, dict]:
    rows = {}
    for q in questions:
        gold = list(dict.fromkeys(q.gold))
        eligible = _eligible(q, probs[q.question_id])
        sent = prepared[q.question_id]["doc_ids"]
        if any(doc_id not in eligible for doc_id in sent):
            raise RuntimeError(f"{q.question_id}: a sent document was not kept by the filter.")
        cand_ids = [c.doc_id for c in q.candidates]
        rows[q.question_id] = {
            "gold": gold,
            "in_candidates": [doc_id for doc_id in gold if doc_id in cand_ids],
            "kept": [doc_id for doc_id in gold if doc_id in eligible],
            "in_prompt": [doc_id for doc_id in gold if doc_id in sent],
            "docs_kept": len(sent),
            "docs_cut": len([doc_id for doc_id in eligible if doc_id not in sent]),
            "prompt_tokens": _prompt_tokens(prepared[q.question_id]),
            "floor": prepared[q.question_id]["floor"],
        }
    return rows


async def _answer(prepared: dict[str, dict], wanted: set[str]) -> int:
    path = OUT_DIR / "answers.jsonl"
    if path.is_file():
        have = {row["question_id"] for row in load_answers(path)}
        if have == wanted:
            logger.info("Parked answers already on disk.")
            return 0
    by_text = {item["question"]: qid for qid, item in prepared.items() if qid in wanted}
    if len(by_text) != len(wanted):
        raise RuntimeError("Two parked questions share the same text.")
    llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens)

    async def answer(question: str) -> dict:
        item = prepared[by_text[question]]
        context = "\n\n".join(p["text"] for p in item["passages"])
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
        dry_run=False, evaluate=False, concurrency=5, only_empty=False)
    code = await _run_phase("p2_parked_baseline", answer, args, OUT_DIR, "cascade", wanted)
    print("\n=== parked answering cost ===")
    print((OUT_DIR / "answer_cost.json").read_text())
    return code


def _as_score(raw: dict) -> QuestionScore:
    return QuestionScore(**{k: raw[k] for k in _FIELDS if k in raw})


def _load_eval(path: Path) -> dict[str, QuestionScore]:
    return {row.question_id: row for row in (_as_score(json.loads(line)) for line in path.open(encoding="utf-8"))}


def _mean(values: list[float]) -> float | None:
    return None if not values else round(sum(values) / len(values), 2)


def _retrieval(ids: list[str], layout: dict[str, dict]) -> dict:
    rows = [layout[qid] for qid in ids]
    found = sum(len(row["in_candidates"]) for row in rows)
    asked = sum(len(row["gold"]) for row in rows)
    return {
        "gold_found": found,
        "gold_asked": asked,
        "gold_found_pct": None if not asked else round(100 * found / asked, 2),
        "docs_kept": _mean([row["docs_kept"] for row in rows]),
        "prompt_tokens": _mean([row["prompt_tokens"] for row in rows]),
        "docs_cut": _mean([row["docs_cut"] for row in rows]),
        "docs_cut_total": sum(row["docs_cut"] for row in rows),
    }


def _block(ids: list[str], scores: dict[str, QuestionScore], layout: dict[str, dict]) -> dict:
    packed = aggregate([scores[qid] for qid in ids])
    return {"judged": packed["overall"], "retrieval": _retrieval(ids, layout)}


def _contradiction(score: QuestionScore) -> bool:
    """Wrong, and the judge's reason is a contradiction rather than a miss."""
    if score.correct:
        return False
    parsed = parse_jev(score.correctness_rationale or "")
    if parsed is not None:
        # parse_jev returns (choice, main_point, contradicts, unused).
        choice, main_point, contradicts, _ = parsed
        return choice == "answers" and main_point >= 0.70 and contradicts >= 0.30
    text = (score.correctness_rationale or "").lower()
    return "contradict" in text or "conflict" in text


def _failure_group(qid: str, cascade: QuestionScore, jev: QuestionScore, layout: dict) -> str | None:
    """First matching cause. A question that scores 1 is not a failure."""
    if cascade.score is not None and cascade.score >= 1:
        return None
    row = layout[qid]
    gold = set(row["gold"])
    if gold - set(row["in_candidates"]):
        return "gold_not_found"
    if gold - set(row["kept"]):
        return "gold_filtered_out"
    if gold - set(row["in_prompt"]):
        return "gold_cut_by_budget"
    if cascade.completeness is None or cascade.completeness < 1:
        return "facts_missing"
    if _contradiction(cascade) or _contradiction(jev):
        return "contradiction"
    return "other_wrong"


def _examples(group: str, ids: list[str], qa, answers: dict[str, dict], scores: dict[str, QuestionScore]) -> list[dict]:
    picked = []
    for qid in ids[:3]:
        row = qa.loc[qid]
        picked.append({
            "question_id": qid,
            "question": row.question,
            "gold_answer": row.gold_answer,
            "answer": answers[qid]["answer"],
            "correct": scores[qid].correct,
            "completeness": scores[qid].completeness,
            "rationale": scores[qid].correctness_rationale,
        })
    return picked


def _report(questions, layout: dict[str, dict]) -> dict:
    cascade = _load_eval(OUT_DIR / "cascade" / "answers_eval.jsonl")
    jev = _load_eval(OUT_DIR / "jev" / "answers_eval.jsonl")
    ids = [q.question_id for q in questions]
    if set(ids) - set(cascade) or set(ids) - set(jev):
        raise RuntimeError("A parked question is missing a judge score.")
    qa = load_qa()
    answers = {row["question_id"]: row for row in load_answers(OUT_DIR / "answers.jsonl")}
    per_question = []
    groups: dict[str, list[str]] = defaultdict(list)
    for qid in ids:
        row = layout[qid]
        facts = list(qa.loc[qid].answer_facts)
        missing = cascade[qid].unsupported_facts or []
        present = None if cascade[qid].completeness is None else len(facts) - len(missing)
        group = _failure_group(qid, cascade[qid], jev[qid], layout)
        if group:
            groups[group].append(qid)
        per_question.append({
            "question_id": qid,
            "question_type": cascade[qid].question_type,
            "work": WORK[qid],
            "n_gold": len(row["gold"]),
            "gold_in_candidates": len(row["in_candidates"]),
            "gold_kept": len(row["kept"]),
            "gold_in_prompt": len(row["in_prompt"]),
            "facts_present": present,
            "n_facts": len(facts),
            "failure": group,
        })
    order = ("gold_not_found", "gold_filtered_out", "gold_cut_by_budget", "facts_missing", "contradiction", "other_wrong")
    body = {
        "pipeline": "phase 2.3 fused top 30, jev filter, phase 1 prompt with floor line",
        "by_work": {
            name: {
                "cascade": _block([qid for qid in ids if WORK[qid] == name], cascade, layout),
                "jev": _block([qid for qid in ids if WORK[qid] == name], jev, layout)["judged"],
            }
            for name in ("multi-part", "count/list")
        },
        "by_type": {
            name: {
                "cascade": _block([qid for qid in ids if cascade[qid].question_type == name], cascade, layout),
                "jev": _block([qid for qid in ids if jev[qid].question_type == name], jev, layout)["judged"],
            }
            for name in PARKED
        },
        "overall": {
            "cascade": _block(ids, cascade, layout),
            "jev": _block(ids, jev, layout)["judged"],
        },
        "questions": per_question,
        "failures": {
            name: {"n": len(groups[name]), "examples": _examples(name, groups[name], qa, answers, cascade)}
            for name in order if groups[name]
        },
    }
    REPORT_PATH.write_text(json.dumps(body, indent=2, ensure_ascii=False))
    _print(body)
    return body


def _cell(block: dict, key: str) -> str:
    value = block.get(key)
    if value is None:
        return f"{'—':>8s}"
    return f"{value:8.2f}"


def _print_slice(label: str, cascade: dict, jev: dict) -> None:
    judged = cascade["judged"]
    other = jev
    ret = cascade["retrieval"]
    print(f"\n{label}  n={judged['n']}")
    print(f"{'':22s}{'overall':>8s}{'correct':>8s}{'complete':>8s}{'recall':>8s}{'words':>8s}")
    print(f"{'cascade':22s}{_cell(judged, 'overall_score')}{_cell(judged, 'correctness')}"
          f"{_cell(judged, 'completeness')}{_cell(judged, 'recall_at_k')}{_cell(judged, 'avg_answer_words')}")
    print(f"{'jev only':22s}{_cell(other, 'overall_score')}{_cell(other, 'correctness')}"
          f"{_cell(other, 'completeness')}{_cell(other, 'recall_at_k')}{_cell(other, 'avg_answer_words')}")
    print(f"gold in fused top 30: {ret['gold_found']}/{ret['gold_asked']} ({ret['gold_found_pct']}%)")
    print(f"docs kept {ret['docs_kept']}   prompt tokens {ret['prompt_tokens']}   "
          f"docs cut by budget {ret['docs_cut']} (total {ret['docs_cut_total']})")


def _print(body: dict) -> None:
    print("\n=== parked baseline. Phase 2.3 + floor line. Cascade, with Jev beside it. ===")
    _print_slice("overall", body["overall"]["cascade"], body["overall"]["jev"])
    for name, block in body["by_work"].items():
        _print_slice(name, block["cascade"], block["jev"])
    for name, block in body["by_type"].items():
        _print_slice(name, block["cascade"], block["jev"])
    print("\nfailures")
    for name, block in body["failures"].items():
        print(f"  {name}: {block['n']}")


async def _judge(mode: str, *, dry_run: bool) -> int:
    usage.reset()
    args = parse_args([
        str(OUT_DIR / "answers.jsonl"),
        "--output-dir", str(OUT_DIR / mode),
        "--judge", mode,
    ])
    args.dry_run = dry_run
    return await evaluate_file(args)


async def main_async(dry_run: bool) -> int:
    sources = _parked()
    questions = _fuse(sources)
    index = DocIndex()
    print("\n=== parked answering dry-run upper bound ===")
    print("Context is every passage that fits in the 10k budget, before the relevance cutoff. "
          "Output assumes the full 4,000-token cap. The floor line is included.")
    print(json.dumps(_answer_bound(questions, index), indent=2))

    usage.reset()
    await score_passages(questions, dry_run=True)
    jev_bound = usage.report()
    model = next(iter(jev_bound["models"].values()))
    local = jev_bound.get("total_estimated_cost_usd") or 0
    print("\n=== parked Jev scoring dry-run (cached calls are free) ===")
    print(json.dumps(jev_bound, indent=2))
    print(f"calibrated uncached cost ${round(local * BILLED_TOKEN_RATIO, 4)} "
          f"(local tokens x {BILLED_TOKEN_RATIO}); "
          f"{model.get('cached_calls', 0)} cache hits, {model.get('estimated_calls', 0)} new calls")
    if dry_run:
        print("\nJudge dry-run waits until the answers exist. Cascade prices sol on every "
              "question when Jev has not answered yet, so that figure is an upper bound.")
        return 0

    print("\n=== starting paid Jev scoring, then answering ===")
    usage.reset()
    probs = _load_scores(questions)
    if probs is None:
        probs = await score_passages(questions, dry_run=False)
        if any(p != p for values in probs.values() for p in values):
            raise RuntimeError("Jev scoring left a missing probability.")
        _save_scores(questions, probs)
        (OUT_DIR / "jev_cost.json").write_text(json.dumps(usage.report(), indent=2))
    else:
        logger.info("Loaded saved Jev scores.")
    prepared = prepare(questions, probs, index)
    wanted = {q.question_id for q in questions}
    code = await _answer(prepared, wanted)
    if code != 0:
        return code

    print("\n=== cascade judge dry-run upper bound ===")
    print("Jev has not been consulted yet in this dry run, so sol is priced on every question.")
    code = await _judge("cascade", dry_run=True)
    if code != 0:
        return code
    print("\n=== judging for real: cascade, then Jev only on the same answers ===")
    code = await _judge("cascade", dry_run=False)
    if code != 0:
        return code
    code = await _judge("jev", dry_run=False)
    if code != 0:
        return code
    _report(questions, _layout(questions, probs, prepared))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 final baseline on the 60 parked questions.")
    parser.add_argument("--dry-run", action="store_true", help="Price Jev scoring and answering, then stop.")
    args = parser.parse_args(argv)
    setup_logging()
    return asyncio.run(main_async(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
