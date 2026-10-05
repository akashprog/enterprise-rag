"""One HOLDOUT confirmation of the Phase 2 pipeline. Not a tuning run.

The pipeline is the one adopted on DEV: a hypothetical passage used only for
search, hybrid fusion to 30 documents, the Jev filter, then the Phase 1 prompt
with the floor line. Nothing here is changed after the scores are in.

Usage:
    python -m phase_2_data_mastery.holdout_confirm --dry-run
    python -m phase_2_data_mastery.holdout_confirm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.query_naive import SYSTEM_PROMPT  # noqa: E402
from phase_2_data_mastery.answer_prompt_v2 import PHASE23_DIR, load_split, prepare  # noqa: E402
from phase_2_data_mastery.expand_answer import (  # noqa: E402
    OUT_DIR as DEV_DIR,
    _answer_tokens,
    _build,
    assert_probe_excluded,
)
from phase_2_data_mastery.floor_line import PROMPT_VERSION as FLOOR_VERSION, user_prompt  # noqa: E402
from phase_2_data_mastery.hybrid_candidates import (  # noqa: E402
    ANSWER_PROMPT_VERSION,
    FOCUS,
    load_fused,
)
from phase_2_data_mastery.query_expansion import _generate  # noqa: E402
from phase_2_data_mastery.query_small_to_big import DocIndex, messages  # noqa: E402
from phase_2_data_mastery.relevance_filter import (  # noqa: E402
    BILLED_TOKEN_RATIO,
    load_candidates,
    score_passages,
)
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import QuestionScore, aggregate  # noqa: E402
from shared_utils.llm import CachedChat, usage  # noqa: E402
from shared_utils.runner import _run_phase  # noqa: E402

logger = logging.getLogger("phase2.holdout")

OUT_DIR = settings.paths.results_dir / "phase_2" / "p2_expand_passage_holdout"
PROBES_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "query_expansion_holdout.json"
REPORT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "holdout_confirm.json"
SCORES_PATH = OUT_DIR / "scores.jsonl"
_FIELDS = {f.name for f in fields(QuestionScore)}


def _holdout_questions():
    dev, hold = load_split()
    by_id = {q.question_id: q for q in load_candidates()}
    missing = [qid for qid in hold if qid not in by_id]
    if len(hold) != 210 or missing or set(hold) & set(dev):
        raise RuntimeError(f"HOLDOUT split is not 210 questions disjoint from DEV ({len(missing)} missing).")
    questions = [by_id[qid] for qid in hold]
    bad = [q.question_id for q in questions if q.question_type not in FOCUS]
    if bad:
        raise RuntimeError(f"HOLDOUT contains non-focus questions: {bad[:5]}")
    return dev, hold, questions


def _as_score(raw: dict) -> QuestionScore:
    return QuestionScore(**{k: raw[k] for k in _FIELDS if k in raw})


def _load_eval(path: Path) -> dict[str, QuestionScore]:
    return {row.question_id: row for row in (_as_score(json.loads(line)) for line in path.open(encoding="utf-8"))}


async def _probes(questions, *, generate: bool) -> tuple[dict[str, str] | None, dict, float | None]:
    if PROBES_PATH.is_file():
        raw = json.loads(PROBES_PATH.read_text())
        rows = {row["question_id"]: row["hypothetical_passage"] for row in raw["questions"]}
        if set(rows) == {q.question_id for q in questions}:
            logger.info("HOLDOUT hypothetical passages loaded from disk.")
            return rows, raw.get("cost") or {}, raw.get("seconds")
    _, dry = await _generate(questions, dry_run=True)
    if not generate:
        return None, dry, None
    started = time.perf_counter()
    generated, cost = await _generate(questions, dry_run=False)
    elapsed = round(time.perf_counter() - started, 1)
    PROBES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROBES_PATH.write_text(json.dumps({
        "questions": [
            {"question_id": qid, "hypothetical_passage": row["hypothetical_passage"]}
            for qid, row in generated.items()
        ],
        "cost": cost,
        "seconds": elapsed,
    }, indent=2, ensure_ascii=False))
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


async def _answer(prepared: dict[str, dict], probes: dict[str, str], wanted: set[str]) -> int:
    by_text = {item["question"]: qid for qid, item in prepared.items() if qid in wanted}
    if len(by_text) != len(wanted):
        raise RuntimeError("Two HOLDOUT questions share the same text.")
    llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens)

    async def answer(question: str) -> dict:
        item = prepared[by_text[question]]
        texts = [passage["text"] for passage in item["passages"]]
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
    code = await _run_phase("p2_expand_passage_holdout", answer, args, OUT_DIR, "jev", wanted)
    print("\n=== HOLDOUT answering cost ===")
    print((OUT_DIR / "answer_cost.json").read_text())
    metrics = OUT_DIR / "jev" / "answers_metrics.json"
    if metrics.is_file():
        print(f"judge cost ${json.loads(metrics.read_text()).get('judge_cost_usd')}")
    return code


def _flips(before: dict[str, QuestionScore], after: dict[str, QuestionScore], ids: list[str]) -> dict:
    counts = {"wrong_to_right": 0, "right_to_wrong": 0, "both_right": 0, "both_wrong": 0}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"wrong_to_right": 0, "right_to_wrong": 0})
    for qid in ids:
        old, new = bool(before[qid].correct), bool(after[qid].correct)
        if not old and new:
            kind = "wrong_to_right"
        elif old and not new:
            kind = "right_to_wrong"
        elif new:
            kind = "both_right"
        else:
            kind = "both_wrong"
        counts[kind] += 1
        if kind in ("wrong_to_right", "right_to_wrong"):
            by_type[after[qid].question_type][kind] += 1
    return {"counts": counts, "by_type": by_type}


def _side(scores: list[QuestionScore]) -> dict:
    packed = aggregate(scores)
    return {"overall": packed["overall"], "by_type": packed["by_type"]}


def _report(dev_ids: list[str], hold_ids: list[str]) -> dict:
    phase23 = _load_eval(PHASE23_DIR / "jev" / "answers_eval.jsonl")
    dev_final = _load_eval(DEV_DIR / "jev" / "answers_eval.jsonl")
    hold_final = _load_eval(OUT_DIR / "jev" / "answers_eval.jsonl")
    if set(hold_ids) - set(hold_final):
        raise RuntimeError("HOLDOUT eval is missing questions.")
    if set(dev_ids) - set(dev_final) or set(dev_ids) - set(phase23) or set(hold_ids) - set(phase23):
        raise RuntimeError("A focus question is missing from a saved eval.")
    hold_old = _side([phase23[qid] for qid in hold_ids])
    hold_new = _side([hold_final[qid] for qid in hold_ids])
    focus_ids = list(dev_ids) + list(hold_ids)
    focus_old = _side([phase23[qid] for qid in focus_ids])
    focus_new = _side([dev_final[qid] for qid in dev_ids] + [hold_final[qid] for qid in hold_ids])
    if len(focus_ids) != 420 or len(set(focus_ids)) != 420:
        raise RuntimeError(f"Focus set is not 420 unique questions ({len(focus_ids)}).")
    body = {
        "pipeline": "expansion, hybrid fused top 30, jev filter, phase 1 prompt with floor line",
        "holdout": {
            "phase23": hold_old,
            "final": hold_new,
            "flips": _flips(phase23, hold_final, hold_ids),
        },
        "focus": {
            "phase23": focus_old,
            "final": focus_new,
            "flips": _flips(phase23, {**{qid: dev_final[qid] for qid in dev_ids}, **hold_final}, focus_ids),
        },
    }
    answer_cost = json.loads((OUT_DIR / "answer_cost.json").read_text())
    judge = json.loads((OUT_DIR / "jev" / "answers_metrics.json").read_text())
    jev_cost = json.loads((OUT_DIR / "jev_cost.json").read_text()) if (OUT_DIR / "jev_cost.json").is_file() else {}
    body["costs"] = {
        "jev_usd": jev_cost.get("total_cost_usd"),
        "answer_usd": answer_cost.get("total_cost_usd"),
        "judge_usd": judge.get("judge_cost_usd"),
        "answer_seconds": answer_cost.get("seconds"),
        "answer_failures": answer_cost.get("failures"),
    }
    REPORT_PATH.write_text(json.dumps(body, indent=2))
    _print(body)
    return body


def _cell(block: dict, key: str, spec: str = ".2f") -> str:
    value = block.get(key)
    if value is None:
        return f"{'—':>8s}"
    return f"{value:8{spec}}"


def _line(label: str, block: dict) -> None:
    print(f"{label:28s}{_cell(block, 'overall_score')}{_cell(block, 'correctness')}"
          f"{_cell(block, 'completeness')}{_cell(block, 'recall_at_k')}"
          f"{_cell(block, 'invalid_extra_docs')}{_cell(block, 'avg_answer_words', '.1f')}")


def _print_pair(title: str, old: dict, new: dict, flips: dict) -> None:
    print(f"\n{title}")
    print(f"{'':28s}{'overall':>8s}{'correct':>8s}{'complete':>8s}{'recall':>8s}{'invalid':>8s}{'words':>8s}")
    _line("Phase 2.3", old["overall"])
    _line("final pipeline", new["overall"])
    counts = flips["counts"]
    print(f"flips: {counts['wrong_to_right']} wrong->right, {counts['right_to_wrong']} right->wrong")
    print(f"{'type':28s}{'p23':>8s}{'final':>8s}{'w->r':>6s}{'r->w':>6s}")
    for qtype in FOCUS:
        base = old["by_type"].get(qtype, {})
        here = new["by_type"].get(qtype, {})
        kind = flips["by_type"].get(qtype, {})
        print(f"{qtype:28s}{base.get('overall_score', 0):8.2f}{here.get('overall_score', 0):8.2f}"
              f"{kind.get('wrong_to_right', 0):6d}{kind.get('right_to_wrong', 0):6d}")
        print(f"{'':28s}{_cell(base, 'correctness')}{_cell(here, 'correctness')}"
              f"{_cell(base, 'completeness')}{_cell(here, 'completeness')}"
              f"{_cell(base, 'recall_at_k')}{_cell(here, 'recall_at_k')}"
              f"{_cell(base, 'invalid_extra_docs')}{_cell(here, 'invalid_extra_docs')}"
              f"{_cell(base, 'avg_answer_words', '.1f')}{_cell(here, 'avg_answer_words', '.1f')}")


def _print(body: dict) -> None:
    print("\n=== HOLDOUT confirmation. Final pipeline vs Phase 2.3. Jev. One run. ===")
    costs = body["costs"]
    print(f"Jev ${costs['jev_usd']}   answer ${costs['answer_usd']}   judge ${costs['judge_usd']}")
    _print_pair("HOLDOUT (210), against Phase 2.3 80.97", body["holdout"]["phase23"], body["holdout"]["final"], body["holdout"]["flips"])
    _print_pair("Focus set (420), against Phase 2.3 79.23", body["focus"]["phase23"], body["focus"]["final"], body["focus"]["flips"])


async def main_async(dry_run: bool) -> int:
    dev_ids, hold_ids, sources = _holdout_questions()
    probes, _, _ = await _probes(sources, generate=not dry_run)
    if probes is None:
        print("\nStopping after the expansion dry-run. Jev and answering need the passages.")
        return 0
    questions = _build(sources, probes, None)
    index = DocIndex()
    answer_bound = _answer_tokens(questions, probes, index)
    print("\n=== HOLDOUT answering dry-run upper bound ===")
    print("Context is every passage that fits in the 10k budget, before the relevance cutoff. "
          "Output assumes the full 4,000-token cap. The floor line is included.")
    print(json.dumps(answer_bound, indent=2))
    usage.reset()
    await score_passages(questions, dry_run=True)
    jev_bound = usage.report()
    model = next(iter(jev_bound["models"].values()))
    local = jev_bound.get("total_estimated_cost_usd") or 0
    print("\n=== HOLDOUT Jev scoring dry-run (cached calls are free) ===")
    print(json.dumps(jev_bound, indent=2))
    print(f"calibrated uncached cost ${round(local * BILLED_TOKEN_RATIO, 4)} "
          f"(local tokens x {BILLED_TOKEN_RATIO}); "
          f"{model.get('cached_calls', 0)} cache hits, {model.get('estimated_calls', 0)} new calls")
    if dry_run:
        return 0

    print("\n=== starting paid HOLDOUT scoring, then answering. No further changes. ===")
    usage.reset()
    probs = _load_scores(questions)
    if probs is None:
        probs = await score_passages(questions, dry_run=False)
        if any(p != p for values in probs.values() for p in values):
            raise RuntimeError("Jev scoring left a missing probability.")
        _save_scores(questions, probs)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "jev_cost.json").write_text(json.dumps(usage.report(), indent=2))
    prepared = prepare(questions, probs, index)
    for q in questions:
        texts = [p["text"] for p in prepared[q.question_id]["passages"]]
        assert_probe_excluded(probes[q.question_id], q.question, texts, "the answer prompt")
    code = await _answer(prepared, probes, set(hold_ids))
    if code != 0:
        return code
    _report(dev_ids, hold_ids)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Confirm the Phase 2 pipeline on HOLDOUT once.")
    parser.add_argument("--dry-run", action="store_true", help="Price the calls and stop.")
    args = parser.parse_args(argv)
    setup_logging()
    return asyncio.run(main_async(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
