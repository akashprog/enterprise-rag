"""Can Jev replace gpt-5.6-sol as our judge? Measure it against verdicts we already paid for.

Why this experiment exists (cost):
    Judging one full 500-question run with gpt-5.6-sol costs ~$4 -- over 10x the
    cost of *answering* the questions. Jev (TypeSafe System One) bills ~$0.042 per
    1M input tokens with free output, i.e. roughly 100-200x cheaper. Before trusting
    it with our scores we need evidence that it agrees with sol. We already have
    that ground truth on disk: the easy-corpus Phase 1 evaluation stored sol's
    correctness verdict per question and the list of facts it found unsupported.
    So the whole experiment costs only the Jev calls (~3 cents) and zero OpenAI calls.

What Jev is asked, and how its answers become verdicts, lives in
`shared_utils/jev_judge.py` -- the same code the evaluator's cascade judge uses,
so this experiment always measures exactly what `--judge cascade` would do.

What is reported (results/judge_comparison/):
    * Agreement and Cohen's kappa vs sol, for correctness and per fact, by question
      type, at the production thresholds of jev_judge.py.
      Kappa corrects for chance: 0.6-0.8 is "substantial", >0.8 "almost perfect".
    * The Overall Score (mean correct x completeness) each judge would report --
      what matters most, since phases are compared on it.
    * A sweep with one shared threshold for every Noul, for comparison.
    * The cascade (Jev first, uncertain verdicts to sol): how many verdicts escalate,
      what agreement remains, and what it would cost.
    * The largest disagreements side by side, to see *why* the judges differ.

Caveat: the production thresholds and bands were tuned on these same 485
questions, so the numbers are optimistic. Re-run on a new sol-judged set (e.g. the
harder corpus) to check they hold.

Usage:
    python scripts/compare_jev_judge.py --dry-run      # token + cost estimate, no calls
    python scripts/compare_jev_judge.py --limit 20     # smoke test
    python scripts/compare_jev_judge.py                # all judged questions

Reads:  results/phase_1/easy/answers.jsonl, results/phase_1/easy/sol/answers_eval.jsonl,
        data/raw_onyx_subset/mini_redwood_qa.jsonl, .cache/api_cache.sqlite
Writes: results/judge/comparison/{metrics.json, verdicts.jsonl, disagreements.jsonl}
        and Jev responses into the shared API cache (re-runs are free).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_utils import jev_judge as jj  # noqa: E402
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import load_answers, load_qa, strip_citations  # noqa: E402
from shared_utils.llm import run_limited, usage  # noqa: E402

logger = logging.getLogger("compare_jev_judge")

SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7]

# sol's average cost per question, measured from the cached easy-corpus run
# (correctness: $1.877 / 485, batched completeness: $2.166 / 485). Used only to
# price the cascade estimate.
SOL_COST_PER_CORRECTNESS = 1.877 / 485
SOL_COST_PER_COMPLETENESS = 2.166 / 485


# =============================================================================
# Data: join sol verdicts, answers and QA
# =============================================================================
def load_cases(answers_path: Path, eval_path: Path) -> list[dict[str, Any]]:
    """One row per question that sol actually judged, with its per-fact labels.

    sol's per-fact verdicts are recovered from `unsupported_facts` (the facts it
    marked unsupported): every other gold fact was judged supported. Empty
    answers are skipped -- the evaluator scores them 0 without calling any judge,
    so there is no sol verdict to compare against. Rows already judged by Jev in
    a cascade run are skipped too: they are not sol ground truth.
    """
    qa = load_qa()
    answers = {a["question_id"]: a for a in load_answers(answers_path)}
    cases = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("correct") is None or ev.get("correctness_rationale") == "empty answer":
                continue
            if ev.get("correctness_judge", "sol") != "sol" or ev.get("completeness_judge", "sol") not in ("sol", "none"):
                continue
            qid = ev["question_id"]
            q = qa.loc[qid]
            facts = list(q["answer_facts"])
            unsupported = set(ev.get("unsupported_facts") or [])
            cases.append({
                "question_id": qid,
                "question_type": q["question_type"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "answer": strip_citations(answers[qid].get("answer") or ""),
                "facts": facts,
                "sol_correct": bool(ev["correct"]),
                "sol_facts": [f not in unsupported for f in facts],
                "sol_completeness": float(ev["completeness"]),
                "sol_score": float(ev["score"]),
                "sol_rationale": ev.get("correctness_rationale"),
            })
    return cases


async def judge_case(judge: jj.JevJudge, case: dict[str, Any]) -> dict[str, Any] | None:
    """Run both Jev calls for one question and return the raw probabilities."""
    sig, probs = await asyncio.gather(
        judge.correctness(case["question"], case["gold_answer"], case["answer"]),
        judge.fact_probs(case["question"], case["answer"], case["facts"]) if case["facts"] else asyncio.sleep(0, []),
    )
    if sig is None or probs is None:
        return None
    return {
        **case,
        "refusal_choice": sig.refusal_choice,
        "refusal_confidence": sig.refusal_confidence,
        "p_declines": sig.p_declines,
        "p_main_point": sig.p_main_point,
        "p_contradicts": sig.p_contradicts,
        "p_facts": probs,
    }


def signals(r: dict[str, Any]) -> jj.CorrectnessSignals:
    return jj.CorrectnessSignals(r["refusal_choice"], r["refusal_confidence"], r["p_declines"],
                                 r["p_main_point"], r["p_contradicts"])


# =============================================================================
# Comparing with sol
# =============================================================================
def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa: agreement beyond what two raters would reach by chance."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def agreement(a: list[bool], b: list[bool]) -> dict[str, Any]:
    """Agreement rate, kappa, and the confusion counts (jev verdict vs sol label)."""
    n = len(a)
    return {
        "n": n,
        "agreement_pct": round(100 * sum(x == y for x, y in zip(a, b)) / n, 2) if n else None,
        "kappa": kappa(a, b),
        "jev_yes_sol_no": sum(x and not y for x, y in zip(a, b)),
        "jev_no_sol_yes": sum(y and not x for x, y in zip(a, b)),
    }


def report_at(rows: list[dict[str, Any]], main_min: float, contra_max: float, fact_min: float) -> dict[str, Any]:
    """Jev-only verdicts at the given thresholds vs sol, overall and per question type."""

    def headline(sub: list[dict[str, Any]]) -> dict[str, Any]:
        jc = [jj.is_correct(signals(r), r["question_type"], main_min, contra_max) for r in sub]
        sc = [r["sol_correct"] for r in sub]
        jf = [p >= fact_min for r in sub for p in r["p_facts"]]
        sf = [s for r in sub for s in r["sol_facts"]]
        jcomp = [sum(p >= fact_min for p in r["p_facts"]) / len(r["p_facts"]) if r["p_facts"] else 1.0 for r in sub]
        jscore = [float(c) * m for c, m in zip(jc, jcomp)]
        mean = lambda xs: round(100 * sum(xs) / len(xs), 2)  # noqa: E731
        return {
            "n": len(sub),
            "correctness": agreement(jc, sc),
            "facts": agreement(jf, sf),
            "overall_score_jev": mean(jscore),
            "overall_score_sol": mean([r["sol_score"] for r in sub]),
            "correctness_pct_jev": mean(jc),
            "correctness_pct_sol": mean(sc),
            "completeness_pct_jev": mean(jcomp),
            "completeness_pct_sol": mean([r["sol_completeness"] for r in sub]),
        }

    types = sorted({r["question_type"] for r in rows})
    return {
        "thresholds": {"main_point_min": main_min, "contradicts_max": contra_max, "fact_min": fact_min},
        "overall": headline(rows),
        "by_type": {t: headline([r for r in rows if r["question_type"] == t]) for t in types},
    }


def cascade(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Simulate `--judge cascade` with the production rules of jev_judge.py.

    An escalated verdict is sol's own verdict, so agreement only drops on
    verdicts Jev kept. Correctness and completeness escalate independently.
    """
    corr_esc = comp_esc = corr_agree = fact_agree = n_facts = 0
    jev_scores, sol_scores = [], []
    for r in rows:
        sig = signals(r)
        c_esc = jj.correctness_uncertain(sig, r["question_type"])
        f_esc = jj.facts_uncertain(r["p_facts"])
        correct = r["sol_correct"] if c_esc else jj.is_correct(sig, r["question_type"])
        if f_esc or not r["p_facts"]:
            completeness, fact_ok = r["sol_completeness"], r["sol_facts"]
        else:
            fact_ok = [p >= jj.FACT_MIN for p in r["p_facts"]]
            completeness = sum(fact_ok) / len(fact_ok)
        corr_esc += c_esc
        comp_esc += f_esc
        corr_agree += correct == r["sol_correct"]
        fact_agree += sum(a == b for a, b in zip(fact_ok, r["sol_facts"]))
        n_facts += len(r["sol_facts"])
        jev_scores.append(float(correct) * completeness)
        sol_scores.append(r["sol_score"])

    n = len(rows)
    sol_cost = corr_esc * SOL_COST_PER_CORRECTNESS + comp_esc * SOL_COST_PER_COMPLETENESS
    return {
        "rules": {"main_point_band": jj.MAIN_POINT_BAND, "contradicts_band": jj.CONTRADICTS_BAND,
                  "fact_band": jj.FACT_BAND, "refusal_confidence_min": jj.REFUSAL_CONFIDENCE_MIN},
        "correctness_escalated_pct": round(100 * corr_esc / n, 2),
        "completeness_escalated_pct": round(100 * comp_esc / n, 2),
        "correctness_agreement_pct": round(100 * corr_agree / n, 2),
        "fact_agreement_pct": round(100 * fact_agree / n_facts, 2) if n_facts else None,
        "overall_score_cascade": round(100 * sum(jev_scores) / n, 2),
        "overall_score_sol": round(100 * sum(sol_scores) / n, 2),
        "sol_cost_usd_for_escalations": round(sol_cost, 3),
        "sol_cost_usd_full_run": round(n * (SOL_COST_PER_CORRECTNESS + SOL_COST_PER_COMPLETENESS), 3),
    }


def disagreements(rows: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    """The `top` Jev-only correctness disagreements where Jev was most confident."""
    out = []
    for r in rows:
        sig = signals(r)
        jc = jj.is_correct(sig, r["question_type"])
        if jc == r["sol_correct"]:
            continue
        deciding = r["p_declines"] if r["question_type"] == jj.NOT_FOUND_TYPE else r["p_main_point"]
        out.append({
            "question_id": r["question_id"],
            "question_type": r["question_type"],
            "sol_correct": r["sol_correct"],
            "jev_correct": jc,
            "escalated_in_cascade": jj.correctness_uncertain(sig, r["question_type"]),
            "margin": round(abs(deciding - 0.5), 3),
            "jev": sig.describe(),
            "sol_rationale": r["sol_rationale"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "answer": r["answer"],
        })
    return sorted(out, key=lambda d: -d["margin"])[:top]


# =============================================================================
# Printing
# =============================================================================
def print_report(rep: dict[str, Any], sweep: list[dict[str, Any]], casc: dict[str, Any], cost_lines: list[str]) -> None:
    o, t = rep["overall"], rep["thresholds"]
    print(f"\n=== Jev only vs gpt-5.6-sol on {o['n']} judged questions "
          f"(main_point >= {t['main_point_min']}, contradicts < {t['contradicts_max']}, fact >= {t['fact_min']}) ===")
    print(f"Correctness agreement: {o['correctness']['agreement_pct']}%  kappa={o['correctness']['kappa']}"
          f"  (jev yes/sol no: {o['correctness']['jev_yes_sol_no']}, jev no/sol yes: {o['correctness']['jev_no_sol_yes']})")
    print(f"Per-fact agreement:    {o['facts']['agreement_pct']}%  kappa={o['facts']['kappa']}"
          f"  over {o['facts']['n']} facts")
    print(f"Overall Score:  jev {o['overall_score_jev']:.2f}  vs  sol {o['overall_score_sol']:.2f}")
    print(f"Correctness %:  jev {o['correctness_pct_jev']:.2f}  vs  sol {o['correctness_pct_sol']:.2f}")
    print(f"Completeness %: jev {o['completeness_pct_jev']:.2f}  vs  sol {o['completeness_pct_sol']:.2f}")

    print("\nBy question type:")
    print(pd.DataFrame([{
        "type": qtype, "n": m["n"],
        "corr_agree": m["correctness"]["agreement_pct"], "corr_kappa": m["correctness"]["kappa"],
        "fact_agree": m["facts"]["agreement_pct"],
        "score_jev": m["overall_score_jev"], "score_sol": m["overall_score_sol"],
    } for qtype, m in rep["by_type"].items()]).set_index("type").to_string(na_rep="-"))

    print("\nOne shared threshold for every Noul (for comparison):")
    print(pd.DataFrame([{
        "t": s["thresholds"]["fact_min"],
        "corr_agree": s["overall"]["correctness"]["agreement_pct"],
        "corr_kappa": s["overall"]["correctness"]["kappa"],
        "fact_agree": s["overall"]["facts"]["agreement_pct"],
        "fact_kappa": s["overall"]["facts"]["kappa"],
        "score_jev": s["overall"]["overall_score_jev"],
    } for s in sweep]).set_index("t").to_string())

    print(f"\nCascade (= --judge cascade), rules {casc['rules']}:")
    print(f"  escalated to sol: correctness {casc['correctness_escalated_pct']}%, "
          f"completeness {casc['completeness_escalated_pct']}%")
    print(f"  agreement with sol: correctness {casc['correctness_agreement_pct']}%, facts {casc['fact_agreement_pct']}%")
    print(f"  Overall Score: cascade {casc['overall_score_cascade']:.2f} vs sol {casc['overall_score_sol']:.2f}")
    print(f"  sol cost: ${casc['sol_cost_usd_for_escalations']} for escalations vs "
          f"${casc['sol_cost_usd_full_run']} for sol on everything")
    for line in cost_lines:
        print(f"[cost] {line}")


# =============================================================================
# Entry point
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    archive = settings.paths.results_dir / "phase_1" / "easy"
    p = argparse.ArgumentParser(description="Compare Jev judge verdicts against cached sol verdicts.")
    p.add_argument("--answers", type=Path, default=archive / "answers.jsonl")
    p.add_argument("--sol-eval", type=Path, default=archive / "sol" / "answers_eval.jsonl",
                   help="Per-question eval JSONL produced with --judge sol.")
    p.add_argument("--limit", type=int, default=None, help="Only the first N judged questions.")
    p.add_argument("--dry-run", action="store_true", help="Estimate Jev tokens/cost without calling it.")
    p.add_argument("--top", type=int, default=15, help="How many disagreements to save.")
    p.add_argument("--output-dir", type=Path, default=settings.paths.results_dir / "judge" / "comparison")
    return p.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.answers, args.sol_eval)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    if args.limit:
        cases = cases[: args.limit]
    if not args.dry_run and not settings.typesafe_api_key:
        logger.error("TYPESAFE_API_KEY is not set. Add it to .env or use --dry-run.")
        return 1
    logger.info("Comparing on %d sol-judged questions (%d facts).",
                len(cases), sum(len(c["facts"]) for c in cases))

    judge = jj.JevJudge(dry_run=args.dry_run)
    try:
        results = await run_limited(cases, lambda c: judge_case(judge, c), desc="jev judging")
    finally:
        await judge.aclose()

    if args.dry_run:
        for line in usage.summary_lines():
            print(f"[dry-run] {line}")
        return 0

    rows = [r for r in results if r is not None]
    rep = report_at(rows, jj.MAIN_POINT_MIN, jj.CONTRADICTS_MAX, jj.FACT_MIN)
    sweep = [report_at(rows, t, t, t) for t in SWEEP]
    casc = cascade(rows)
    print_report(rep, sweep, casc, usage.summary_lines())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "jev_model": settings.jev_model,
        "prompt_version": jj.PROMPT_VERSION,
        "sol_eval": str(args.sol_eval),
        "report": rep,
        "sweep": sweep,
        "cascade": casc,
        "jev_cost": usage.report(),
    }, indent=2))
    with (args.output_dir / "verdicts.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (args.output_dir / "disagreements.jsonl").open("w", encoding="utf-8") as f:
        for d in disagreements(rows, args.top):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("Wrote metrics.json, verdicts.jsonl, disagreements.jsonl to %s", args.output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    try:
        return asyncio.run(run(parse_args(argv)))
    except KeyboardInterrupt:
        logger.error("Interrupted. Jev answers so far are cached; re-run to resume for free.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
