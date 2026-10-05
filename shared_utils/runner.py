"""Run a phase's `answer()` over the benchmark questions, save results, optionally score them.

Every phase exposes the same contract:

    async def answer(question: str) -> {"answer": str, "doc_ids": list[str]}

and calls `run_phase(...)` from its query script. That keeps the expensive,
easy-to-get-wrong parts identical across phases: concurrency limits, error
isolation, output format, cost reporting, and evaluation.

What `run_phase` writes, inside `out_dir` (for Phase 1, results/phase_1/fixed/):
    answers.jsonl                 {question_id, answer, doc_ids} per question
    answer_cost.json              tokens + dollars spent *answering* (not judging)
    <judge>/answers_eval.jsonl, <judge>/answers_metrics.json   (with --evaluate)

Every completed answering run and evaluation is also appended to results/log.jsonl.
The human-readable index of all runs is results/README.md.

Cost notes:
    * Answering and judging costs are reported separately (the usage tracker is
      reset in between), so a phase's "cost" column reflects what the pipeline
      itself would cost in production, not what grading it cost us.
    * A failing question is recorded with an empty answer and the error, and the
      run continues -- one bad request never throws away 499 paid answers.
    * `--only-empty` re-answers just those empty/failed rows later and merges
      them into the existing answers file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from shared_utils import evaluation
from shared_utils.config import settings
from shared_utils.llm import run_limited, usage

logger = logging.getLogger(__name__)

AnswerFn = Callable[[str], Awaitable[dict[str, Any]]]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags every query script shares."""
    parser.add_argument("--limit", type=int, default=None, help="Answer only the first N questions.")
    parser.add_argument("--question-type", default=None,
                        help="Only questions of this type (e.g. conflicting_info).")
    parser.add_argument("--question-types", default=None,
                        help="Only these question types, comma-separated. Combines with --question-type.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate answering cost without calling paid APIs.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Score the answers with shared_utils.evaluation afterwards.")
    parser.add_argument("--concurrency", type=int, default=settings.max_concurrency,
                        help="Max questions answered in parallel.")
    parser.add_argument("--only-empty", action="store_true",
                        help="Re-answer only questions whose answer in the existing answers file is empty "
                             "or failed; keep every other answer as is.")


def run_phase(phase: str, answer_fn: AnswerFn, args: argparse.Namespace,
              *, out_dir: Path | None = None, judge: str | None = None,
              question_ids: set[str] | None = None) -> int:
    """Answer the selected questions, persist answers + cost, optionally evaluate.

    Args:
        phase:     Short tag for logs (e.g. "p1" or a context-setting tag).
        answer_fn: The phase's async `answer(question)` function.
        args:      Parsed CLI args (see `add_common_args`).
        out_dir:   Where this run's files go (default `results/<phase>/`). One
                   directory per run, so arms and judges never overwrite each other.
        judge:     Judge for --evaluate ("jev", "sol" or "cascade"). Defaults to
                   `settings.judge_mode`. Development runs pass "jev".
        question_ids: If set, answer only these questions. Type filters still apply.

    Returns:
        Process exit code.
    """
    try:
        # One event loop for answering *and* judging (see evaluation.evaluate_file).
        return asyncio.run(_run_phase(
            phase, answer_fn, args, out_dir or settings.paths.results_dir / phase, judge, question_ids))
    except KeyboardInterrupt:
        logger.error("Interrupted. Completed API calls are cached; re-run to resume for free.")
        return 130


async def _run_phase(phase: str, answer_fn: AnswerFn, args: argparse.Namespace, out_dir: Path,
                     judge: str | None = None, question_ids: set[str] | None = None) -> int:
    """Async body of `run_phase`."""
    qa = evaluation.load_qa()
    types = evaluation.selected_question_types(args)
    if types:
        qa = qa[qa["question_type"].isin(types)]
    if question_ids is not None:
        missing = question_ids - set(qa["question_id"])
        if missing:
            logger.error("%d question ids are not in the selected QA.", len(missing))
            return 1
        qa = qa[qa["question_id"].isin(question_ids)]
    if args.limit:
        qa = qa.head(args.limit)
    answers_path = out_dir / "answers.jsonl"
    previous: list[dict[str, Any]] = []
    if args.only_empty:
        # Re-answering only what failed is valid because an answer that finished
        # normally ("stop") is unaffected by settings that only matter when hit,
        # such as a higher max_tokens cap. Everything else is kept verbatim.
        if not answers_path.is_file():
            logger.error("--only-empty needs an existing %s.", answers_path)
            return 1
        previous = evaluation.load_answers(answers_path)
        redo = {r["question_id"] for r in previous if not (r.get("answer") or "").strip() or r.get("error")}
        qa = qa[qa["question_id"].isin(redo)]
        logger.info("--only-empty: re-answering %d of %d questions.", len(qa), len(previous))
    if qa.empty:
        logger.error("No questions selected.")
        return 1
    questions = qa.to_dict("records")

    async def one(q: dict[str, Any]) -> dict[str, Any]:
        try:
            out = await answer_fn(q["question"])
            return {"question_id": q["question_id"], "answer": out.get("answer", ""),
                    "doc_ids": list(out.get("doc_ids", []))}
        except Exception as exc:  # noqa: BLE001 - isolate per-question failures
            logger.exception("Question %s failed: %s", q["question_id"], exc)
            return {"question_id": q["question_id"], "answer": "", "doc_ids": [], "error": repr(exc)}

    started = time.perf_counter()
    rows = await run_limited(questions, one, limit=args.concurrency, desc=f"{phase} answering")
    elapsed = time.perf_counter() - started

    failures = sum(1 for r in rows if "error" in r)
    logger.info("Answered %d questions in %.1fs (%d failures).", len(rows), elapsed, failures)
    usage.log_summary()

    if args.dry_run:
        logger.info("--dry-run: no answers written.")
        return 0

    cost = {**usage.report(), "questions": len(rows), "failures": failures, "seconds": round(elapsed, 1)}
    if previous:
        fresh = {r["question_id"]: r for r in rows}
        rows = [fresh.get(r["question_id"], r) for r in previous]
        cost["only_empty_reanswered"] = sorted(fresh)
    evaluation.save_answers(rows, answers_path)
    # In --only-empty mode this file records only the re-answering run's cost.
    cost_path = out_dir / f"answer_cost{'_only_empty' if previous else ''}.json"
    cost_path.write_text(json.dumps(cost, indent=2))
    logger.info("Wrote %s", answers_path)
    evaluation.append_run_log({
        "kind": "answers", "phase": phase, "dir": str(out_dir),
        "questions": len(rows), "failures": failures, "seconds": round(elapsed, 1),
        "only_empty": bool(previous), "cost_usd": cost.get("total_cost_usd"),
        "answer_model": settings.answer_model, "answer_max_tokens": settings.answer_max_tokens,
        "answer_reasoning_effort": settings.answer_reasoning_effort,
    })

    if args.evaluate:
        usage.reset()  # report judge cost separately from answer cost
        # Each judge gets its own subdirectory, so a jev run and a cascade run of
        # the same answers coexist.
        judge_name = judge or settings.judge_mode
        judge_dir = out_dir / judge_name
        return await evaluation.evaluate_file(evaluation.parse_args(
            [str(answers_path), "--output-dir", str(judge_dir), "--judge", judge_name]))
    return 1 if failures == len(rows) else 0
