"""Floor line on the questions Phase 2.3 kept by the floor.

The answer prompt is Phase 2.3's, plus the floor sentence when every passage
sent scored under 0.7 (with a floor of 1, that is exactly one passage). This
combination, Phase 2.3 plus the floor line, is the baseline for later DEV
comparisons. HOLDOUT was not answered. Questions that did not get the line
keep their Phase 2.3 answers and Jev verdicts.

Usage:
    python -m phase_2_data_mastery.floor_line --dry-run
    python -m phase_2_data_mastery.floor_line
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.query_naive import SYSTEM_PROMPT  # noqa: E402
from phase_2_data_mastery.answer_prompt_v2 import (  # noqa: E402
    FLOOR_LINE,
    PHASE23_DIR,
    load_split,
    prepare,
)
from phase_2_data_mastery.hybrid_candidates import GUARD, load_fused  # noqa: E402
from phase_2_data_mastery.query_small_to_big import DocIndex  # noqa: E402
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import QuestionScore, aggregate  # noqa: E402
from shared_utils.llm import CachedChat, usage  # noqa: E402
from shared_utils.runner import _run_phase  # noqa: E402

logger = logging.getLogger("phase2.floor_line")

OUT_DIR = settings.paths.results_dir / "phase_2" / "p2_floor_line_dev"
REPORT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "floor_line_dev.json"
# Distinct from p2-hybrid-v1, so these calls are not served from the Phase 2.3 cache.
PROMPT_VERSION = "p2-hybrid-floor-v1"
_SCORE_FIELDS = {f.name for f in fields(QuestionScore)}


def user_prompt(question: str, context: str) -> str:
    """Phase 2.3's user prompt, with the floor sentence between the question and the context."""
    return f"Question: {question}\n\n{FLOOR_LINE}\n\nContext:\n{context}\n\nAnswer:"


def floor_ids(prepared: dict[str, dict], dev: set[str], guard: set[str]) -> list[str]:
    """Questions on DEV or the guard whose only passage was kept by the floor."""
    chosen = []
    for qid, item in prepared.items():
        if qid not in dev and qid not in guard:
            continue
        if not item["floor"]:
            continue
        if len(item["passages"]) != 1:
            raise RuntimeError(
                f"{qid} is below 0.7 but sent {len(item['passages'])} passages; "
                "the floor line is defined for the single floor passage."
            )
        chosen.append(qid)
    return chosen


def _score(raw: dict) -> QuestionScore:
    return QuestionScore(**{k: raw[k] for k in _SCORE_FIELDS if k in raw})


def _load_eval(path: Path) -> dict[str, QuestionScore]:
    return {row.question_id: row for row in (_score(json.loads(line)) for line in path.open(encoding="utf-8"))}


def _flips(before: dict[str, QuestionScore], after: dict[str, QuestionScore], ids: list[str]) -> dict[str, int]:
    counts = {"wrong_to_right": 0, "right_to_wrong": 0, "both_right": 0, "both_wrong": 0}
    for qid in ids:
        old, new = bool(before[qid].correct), bool(after[qid].correct)
        if not old and new:
            counts["wrong_to_right"] += 1
        elif old and not new:
            counts["right_to_wrong"] += 1
        elif new:
            counts["both_right"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _block(scores: list[QuestionScore]) -> dict:
    return aggregate(scores)["overall"]


def report(dev: list[str], guard: list[str], answered: list[str]) -> dict:
    """Full DEV and the guard, with only the floor questions replaced."""
    phase23 = _load_eval(PHASE23_DIR / "jev" / "answers_eval.jsonl")
    fresh = _load_eval(OUT_DIR / "jev" / "answers_eval.jsonl")
    missing = [qid for qid in answered if qid not in fresh]
    if missing:
        raise RuntimeError(f"Missing new Jev rows for {missing}")
    merged = dict(phase23)
    merged.update(fresh)

    def take(ids: list[str], source: dict[str, QuestionScore]) -> list[QuestionScore]:
        return [source[qid] for qid in ids]

    dev_floor = [qid for qid in answered if qid in set(dev)]
    guard_floor = [qid for qid in answered if qid in set(guard)]
    out = {
        "prompt_version": PROMPT_VERSION,
        "prompt": "phase 2.3, plus the floor line on floor questions only",
        "reanswered": answered,
        "dev": {
            "phase23": _block(take(dev, phase23)),
            "floor_line": _block(take(dev, merged)),
            "flips": _flips(phase23, merged, dev),
            "n": len(dev),
            "reanswered": len(dev_floor),
        },
        "info_not_found": {
            "phase23": _block(take(guard, phase23)),
            "floor_line": _block(take(guard, merged)),
            "flips": _flips(phase23, merged, guard),
            "n": len(guard),
            "reanswered": len(guard_floor),
            "kept_phase23": sorted(set(guard) - set(guard_floor)),
        },
        "floor_subset": {
            "dev": {
                "ids": dev_floor,
                "phase23": _block(take(dev_floor, phase23)) if dev_floor else None,
                "floor_line": _block(take(dev_floor, merged)) if dev_floor else None,
            },
            "info_not_found": {
                "ids": guard_floor,
                "phase23": _block(take(guard_floor, phase23)) if guard_floor else None,
                "floor_line": _block(take(guard_floor, merged)) if guard_floor else None,
            },
        },
    }
    REPORT_PATH.write_text(json.dumps(out, indent=2))
    _print(out)
    return out


def _fmt(row: dict | None, key: str, spec: str = ".2f") -> str:
    if not row or row.get(key) is None:
        return f"{'—':>8s}"
    return f"{row[key]:8{spec}}"


def _line(label: str, row: dict | None) -> None:
    print(f"{label:28s}{_fmt(row, 'overall_score')}{_fmt(row, 'correctness')}"
          f"{_fmt(row, 'completeness')}{_fmt(row, 'avg_answer_words', '.1f')}")


def _print(report_body: dict) -> None:
    print("\n=== Floor line. Phase 2.3 prompt, line added only where the floor kept the passage. Jev ===")
    print(f"{'':28s}{'overall':>8s}{'correct':>8s}{'complete':>8s}{'words':>8s}")
    for name, block in (("DEV", report_body["dev"]), ("info_not_found", report_body["info_not_found"])):
        print(f"\n{name} ({block['n']} questions, {block['reanswered']} re-answered)")
        _line("Phase 2.3", block["phase23"])
        _line("floor line", block["floor_line"])
        flips = block["flips"]
        print(f"flips: {flips['wrong_to_right']} wrong->right, {flips['right_to_wrong']} right->wrong")
    subset = report_body["floor_subset"]
    print("\nRe-answered subset only")
    for name in ("dev", "info_not_found"):
        block = subset[name]
        print(f"\n{name} ({len(block['ids'])})")
        _line("Phase 2.3", block["phase23"])
        _line("floor line", block["floor_line"])


async def _answer(prepared: dict[str, dict], wanted: set[str], *, dry_run: bool) -> int:
    by_text = {item["question"]: qid for qid, item in prepared.items() if qid in wanted}
    if len(by_text) != len(wanted):
        raise RuntimeError("Two floor questions share the same text; answering keys on the text.")
    llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens, dry_run=dry_run)

    async def answer(question: str) -> dict:
        item = prepared[by_text[question]]
        context = "\n\n".join(p["text"] for p in item["passages"])
        messages = [("system", SYSTEM_PROMPT), ("user", user_prompt(item["question"], context))]
        res = await llm.ainvoke(messages, prompt_version=PROMPT_VERSION)
        return {"answer": res.text, "doc_ids": item["doc_ids"]}

    usage.reset()
    args = argparse.Namespace(
        limit=None, question_type=None, question_types=None,
        dry_run=dry_run, evaluate=not dry_run, concurrency=5, only_empty=False)
    code = await _run_phase("p2_floor_line_dev", answer, args, OUT_DIR, "jev", wanted)
    label = "dry-run upper bound" if dry_run else "answer cost"
    print(f"\n=== floor line {label} ===")
    if dry_run:
        print(json.dumps(usage.report(), indent=2))
    else:
        print((OUT_DIR / "answer_cost.json").read_text())
        metrics = OUT_DIR / "jev" / "answers_metrics.json"
        if metrics.is_file():
            print(f"judge cost ${json.loads(metrics.read_text()).get('judge_cost_usd')}")
    return code


async def main_async(dry_run: bool) -> int:
    dev, holdout = load_split()
    questions, probs = load_fused()
    dev_set, hold_set = set(dev), set(holdout)
    guard = [q.question_id for q in questions if q.question_type in GUARD]
    if len(guard) != 20:
        raise RuntimeError(f"Expected 20 info_not_found questions, got {len(guard)}.")
    prepared = prepare(questions, probs, DocIndex())
    answered = floor_ids(prepared, dev_set, set(guard))
    if set(answered) & hold_set:
        raise RuntimeError("HOLDOUT question selected for the floor line.")
    outside = [qid for qid in answered if qid not in dev_set and qid not in set(guard)]
    if outside:
        raise RuntimeError(f"Floor selection left the DEV+guard scope: {outside}")
    logger.info(
        "Floor line on %d questions (%d DEV, %d info_not_found). Prompt %s.",
        len(answered),
        sum(qid in dev_set for qid in answered),
        sum(qid in set(guard) for qid in answered),
        PROMPT_VERSION,
    )
    if not answered:
        raise RuntimeError("No floor questions on DEV or the guard.")
    code = await _answer(prepared, set(answered), dry_run=True)
    if code != 0 or dry_run:
        return code
    code = await _answer(prepared, set(answered), dry_run=False)
    if code != 0:
        return code
    report(dev, guard, answered)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-answer floor-line questions with the Phase 2.3 prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Price the answer calls and stop.")
    args = parser.parse_args(argv)
    setup_logging()
    return asyncio.run(main_async(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
