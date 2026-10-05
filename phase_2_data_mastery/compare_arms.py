"""Side-by-side A/B report for two small-to-big settings (Step 2.1 fair A/B).

Arms are named by their context-setting tag (see query_small_to_big.ContextSettings.tag),
e.g. `n5_w1_t2000_b8000` (small-to-big) vs `plain_k40_b5400` (Phase 1 recipe, more chunks).
Each arm must already be answered and judged, in the layout run_phase writes:
results/phase_2/<tag>/answers.jsonl and results/phase_2/<tag>/jev/answers_eval.jsonl.

What it reports, overall and per question type, for Phase 1 (fixed) and each arm:
    Overall, Correct, Complete, Recall@10, Invalid extra docs, avg answer words,
    answer cost (USD for the 500 answers, and per question type).
And on the questions where Phase 1 retrieved every gold document (recall@10 = 1):
    correctness per arm, and how many questions flipped wrong->right / right->wrong
    between the two arms.

Answer cost per question is read back from the API cache: each arm's prompt is
rebuilt exactly (same deterministic retrieval and context rule) and its cache
entry holds the billed input/output tokens. That covers every answer, including
ones produced in an earlier, interrupted run, which the per-run cost files do not.

Cost: $0. Cache reads, local Qdrant, cached question embeddings.

Usage:
    python -m phase_2_data_mastery.compare_arms \\
        --arm A=n5_w1_t2000_b8000 --arm B=plain_k40_b5370

Reads:  results/phase_2/<tag>/{answers.jsonl, jev/answers_eval.jsonl},
        results/phase_1/fixed/jev/answers_eval.jsonl,
        results/phase_1/fixed/cascade/answers_eval.jsonl (the recall=1 set)
Writes: results/phase_2/ab/<A>_vs_<B>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_2_data_mastery.query_small_to_big import (  # noqa: E402
    PROMPT_VERSION,
    ContextSettings,
    SmallToBigPipeline,
    messages,
)
from shared_utils.cache import get_cache  # noqa: E402
from shared_utils.config import settings  # noqa: E402
from shared_utils.evaluation import load_qa  # noqa: E402
from shared_utils.llm import CachedChat, UsageTracker, _normalise_messages  # noqa: E402

_TAG_RE = re.compile(r"^(?:n(\d+)_w(\d+)_t(\d+)_b(\d+)|plain_k(\d+)_b(\d+))$")


def settings_from_tag(tag: str) -> ContextSettings:
    """Inverse of ContextSettings.tag. Small-to-big arms always searched 20 chunks."""
    m = _TAG_RE.match(tag)
    if not m:
        raise ValueError(f"Unrecognised setting tag {tag!r}")
    if m.group(5):
        return ContextSettings(top_chunks=int(m.group(5)), budget=int(m.group(6)), plain=True)
    n, w, t, b = map(int, m.group(1, 2, 3, 4))
    return ContextSettings(docs=n, window=w, whole_doc_max=t, budget=b)


def answer_costs(tag: str, qa: pd.DataFrame) -> pd.Series:
    """USD per question for one arm's answers, from the cached responses."""
    pipeline = SmallToBigPipeline(settings_from_tag(tag))
    keyer = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens)
    cache = get_cache()
    costs = {}
    for q in qa.itertuples():
        msgs = _normalise_messages(messages(q.question, pipeline.context(q.question).text))
        hit = cache.get_json(keyer._key(msgs, PROMPT_VERSION, None))
        if hit is None:
            raise RuntimeError(f"{tag}: no cached answer for {q.question_id}; was the arm fully answered?")
        costs[q.question_id] = UsageTracker.price(settings.answer_model, hit["input_tokens"], hit["output_tokens"])
    return pd.Series(costs, name="answer_cost")


def load_eval(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True).set_index("question_id")


def summarise(ev: pd.DataFrame) -> dict[str, Any]:
    pct = lambda s: None if s.dropna().empty else round(100 * s.dropna().astype(float).mean(), 2)  # noqa: E731
    out = {
        "n": len(ev),
        "overall": pct(ev["score"]),
        "correct": pct(ev["correct"]),
        "complete": pct(ev["completeness"]),
        "recall": pct(ev["recall_at_k"]),
        "invalid": None if ev["invalid_extra_docs"].dropna().empty else round(ev["invalid_extra_docs"].mean(), 2),
        "words": round(ev["answer_words"].mean(), 1),
    }
    if "answer_cost" in ev:
        out["cost_usd"] = round(ev["answer_cost"].sum(), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Side-by-side report for two Phase 2 context settings.")
    p.add_argument("--arm", action="append", required=True, metavar="NAME=TAG",
                   help="Arm name and setting tag; give exactly two.")
    p.add_argument("--recall1-eval", type=Path,
                   default=settings.paths.results_dir / "phase_1" / "fixed" / "cascade" / "answers_eval.jsonl",
                   help="Eval whose recall@10 = 1 questions define the subset (Phase 1 fixed).")
    args = p.parse_args(argv)
    arms = dict(a.split("=", 1) for a in args.arm)
    if len(arms) != 2:
        p.error("give exactly two --arm NAME=TAG")

    qa = load_qa()
    results = settings.paths.results_dir
    evals: dict[str, pd.DataFrame] = {}
    p1 = load_eval(results / "phase_1" / "fixed" / "jev" / "answers_eval.jsonl")
    # Phase 1 (fixed) = the main run plus the --only-empty re-answer of its 16 empty answers.
    p1_cost = sum(json.loads(f.read_text())["total_cost_usd"]
                  for f in (results / "phase_1" / "fixed").glob("answer_cost*.json"))
    evals["Phase 1 (fixed)"] = p1
    for name, tag in arms.items():
        ev = load_eval(results / "phase_2" / tag / "jev" / "answers_eval.jsonl")
        evals[f"{name}: {tag}"] = ev.join(answer_costs(tag, qa))

    report: dict[str, Any] = {"arms": arms, "judge": "jev", "overall": {}, "by_type": {}}
    for label, ev in evals.items():
        report["overall"][label] = summarise(ev)
        report["by_type"][label] = {t: summarise(g) for t, g in ev.groupby("question_type")}
    # Phase 1's per-question costs are not cached under one key (two max_tokens
    # values); its run total is known and is reported for the overall row only.
    report["overall"]["Phase 1 (fixed)"]["cost_usd"] = round(p1_cost, 4)

    # Recall=1 subset (Phase 1's), correctness per arm and flips between arms.
    sub = load_eval(args.recall1_eval)
    sub = sub[sub["recall_at_k"] == 1.0].index
    (na, ea), (nb, eb) = [(n, evals[f"{n}: {t}"].loc[sub, "correct"].astype(bool)) for n, t in arms.items()]
    p1c = evals["Phase 1 (fixed)"].loc[sub, "correct"].astype(bool)
    report["recall1"] = {
        "n": len(sub),
        "correct_pct": {"Phase 1 (fixed)": round(100 * p1c.mean(), 1), na: round(100 * ea.mean(), 1),
                        nb: round(100 * eb.mean(), 1)},
        f"{nb}_wrong_{na}_right": int((~eb & ea).sum()),
        f"{nb}_right_{na}_wrong": int((eb & ~ea).sum()),
        "both_right": int((ea & eb).sum()),
        "both_wrong": int((~ea & ~eb).sum()),
    }

    pd.set_option("display.width", 250)
    cols = ["overall", "correct", "complete", "recall", "invalid", "words", "cost_usd"]
    print("=== Overall (judge: jev only) ===")
    print(pd.DataFrame(report["overall"]).T.reindex(columns=cols).to_string())
    for t in sorted(qa["question_type"].unique()):
        print(f"\n--- {t} ---")
        print(pd.DataFrame({lab: report["by_type"][lab][t] for lab in evals}).T.reindex(columns=["n"] + cols).to_string())
    print("\n=== Phase 1 recall@10 = 1 questions ===")
    print(json.dumps(report["recall1"], indent=2))

    out = results / "phase_2" / "ab" / f"{arms[na]}_vs_{arms[nb]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
