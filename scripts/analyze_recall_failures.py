"""Why are answers wrong when retrieval found every gold document?

For Phase 1 questions with recall@10 = 1 but `correct = false`, this replays the
exact context the answer model saw and writes one CSV row per question with:

    * Refusal check   - did the answer say "I don't know" or similar? (regex)
    * Fact presence   - did the chunks actually in the prompt contain the gold
                        facts? (word-overlap per fact; a document can be
                        retrieved while the chunk holding the fact is not)
    * Distractors     - were this question's own distractor documents (tagged by
                        scripts/add_distractors.py) in the prompt?
    * Repeats         - how many prompt chunks were extra chunks of a document
                        already in the prompt?

Cost: $0. Question embeddings come from the cache, Qdrant is local, and no LLM
is called. Context is rebuilt exactly as `phase_1_baseline/query_naive.py` does:
top-10 cosine chunks, then `fit_to_budget` (chunks past the token budget never
reached the model, so they are excluded from every check). `replay_matches`
confirms the replayed retrieval produced the same doc_ids as the original run.

The fact check is lexical, so it is approximate: a fact is counted as present
when >= FACT_PRESENT_MIN of its content words appear in the prompt context.
Paraphrased facts can be missed and coincidental overlap can over-count; use
`fact_coverage_min` / `missing_facts` to inspect borderline rows.

Other contexts (`--contexts`): instead of replaying Phase 1, check the passages
in a contexts file written by another pipeline, e.g.
`phase_2_data_mastery/query_small_to_big.py --contexts-only`. The questions
analysed are still selected by `--eval` (default: Phase 1's), so a new context
is compared with Phase 1 on exactly the same questions. With `--contexts`, a
"chunk" in the columns below means one passage of that file (Phase 2 sends one
passage per document), and the answer-based columns (refusal, answer) still
describe the `--answers` file.

Usage:
    python scripts/analyze_recall_failures.py
    python scripts/analyze_recall_failures.py --include-correct   # also rows that were right, for comparison
    python scripts/analyze_recall_failures.py --include-correct \\
        --contexts results/phase_2/contexts/n5_w1_t2000_b8000/contexts.jsonl \\
        --output results/phase_2/contexts/n5_w1_t2000_b8000/recall1_all.csv

Reads:  results/phase_1/fixed/answers.jsonl, results/phase_1/fixed/cascade/answers_eval.jsonl,
        data/raw_onyx_subset/mini_redwood_{qa,docs}.jsonl, Qdrant `p1_naive` (or --contexts)
Writes: results/analysis/phase1_recall1_wrong.csv (or phase1_recall1_all.csv, or --output)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.query_naive import TOP_K, NaivePipeline  # noqa: E402
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import load_answers, load_qa  # noqa: E402
from shared_utils.llm import fit_to_budget  # noqa: E402
from shared_utils.vectorstore import doc_ids_from  # noqa: E402

logger = logging.getLogger("analyze_recall_failures")

FACT_PRESENT_MIN = 0.8

# Phrases the tutorial prompt's "say that you don't know" instruction produces.
REFUSAL_PATTERNS = [
    r"\bi (?:do not|don't) know\b",
    r"\b(?:do not|don't|does not|doesn't) (?:say|specify|mention|state|include|provide|contain)\b",
    r"\bnot (?:mentioned|specified|stated|provided|included|contained|available|given|found|clear)\b",
    r"\b(?:no|not enough|insufficient) (?:information|details|context|mention)\b",
    r"\b(?:cannot|can't|unable to) (?:determine|find|confirm|tell|answer|identify)\b",
    r"\bthe (?:provided |retrieved )?context (?:does not|doesn't|did not|only)\b",
    r"\bisn't (?:mentioned|specified|stated|provided)\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)

_STOPWORDS = set("""
a an and are as at be been being but by can could did do does for from had has have how if in into is it
its may might must not of on or our per should so such than that the their them then there these they
this those to was we were what when where which while who why will with would you your also via any all
each both only more most other some same up out over under about after before between during without
""".split())
_WORD_RE = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    """Lowercased alphanumeric words minus stopwords and 1-letter tokens."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def fact_coverage(fact: str, context_words: set[str]) -> float:
    words = content_words(fact)
    return 1.0 if not words else len(words & context_words) / len(words)


def load_doc_tags() -> tuple[dict[str, str], dict[str, set[str]]]:
    """doc_id -> selection (gold/noise/distractor/...), question_id -> its distractor doc_ids."""
    docs = pd.read_json(settings.paths.mini_docs, lines=True, dtype={"doc_id": str})
    docs = docs[["doc_id", "selection", "distractor_for"]]
    selection = dict(zip(docs["doc_id"], docs["selection"]))
    distractors: dict[str, set[str]] = defaultdict(set)
    for doc_id, qids in zip(docs["doc_id"], docs["distractor_for"]):
        for q in qids if isinstance(qids, list) else []:
            distractors[q].add(doc_id)
    return selection, distractors


def phase1_passages(question: str, pipeline: NaivePipeline) -> tuple[list[str], list[str], list[str]]:
    """Replay Phase 1's context: (passage texts, their doc_ids, all retrieved doc_ids).

    Passages are the chunks that survived `fit_to_budget`; the third value is
    every retrieved document (what Phase 1 reported), used for `replay_matches`.
    """
    hits = pipeline.store.similarity_search(question, TOP_K)
    kept = fit_to_budget([h.page_content for h in hits])
    return kept, [h.metadata.get("doc_id") for h in hits[: len(kept)]], doc_ids_from(hits)


def load_contexts(path: Path) -> dict[str, dict[str, Any]]:
    """question_id -> contexts-file row ({doc_ids, passages: [{doc_id, text}, ...], ...})."""
    with path.open(encoding="utf-8") as f:
        return {r["question_id"]: r for r in map(json.loads, f) if r}


def analyse(row: dict[str, Any], ev: dict[str, Any], q: pd.Series, passages: tuple[list[str], list[str], int],
            selection: dict[str, str], own_distractors: set[str], replay_matches: bool | None) -> dict[str, Any]:
    """Compute every check for one question.

    Args:
        passages:       (texts in the prompt, their doc_ids, number retrieved before the budget).
        replay_matches: Whether the replayed retrieval equals the original run (None if not replayed).
    """
    kept, context_docs, n_retrieved = passages
    gold = set(q["expected_doc_ids"])

    # Fact presence, against the text the model actually received.
    ctx_words = content_words("\n\n".join(kept))
    facts = list(q["answer_facts"])
    coverages = [fact_coverage(f, ctx_words) for f in facts]
    missing = [f for f, c in zip(facts, coverages) if c < FACT_PRESENT_MIN]
    gold_chunk_words = content_words("\n\n".join(t for t, d in zip(kept, context_docs) if d in gold))
    in_gold_chunks = sum(fact_coverage(f, gold_chunk_words) >= FACT_PRESENT_MIN for f in facts)

    # Refusal.
    answer = row.get("answer") or ""
    refusal = _REFUSAL_RE.search(answer)

    # Context composition and repeats.
    per_doc = Counter(context_docs)
    first_rank = lambda pred: next((i + 1 for i, d in enumerate(context_docs) if pred(d)), None)  # noqa: E731

    return {
        "question_id": q["question_id"],
        "question_type": q["question_type"],
        "correctness_judge": ev.get("correctness_judge"),
        "completeness": ev.get("completeness"),
        "replay_matches": replay_matches,
        # 1. refusal
        "says_dont_know": bool(refusal),
        "refusal_phrase": refusal.group(0) if refusal else "",
        "answer_words": len(answer.split()),
        # 2. gold facts in the prompt
        "n_facts": len(facts),
        "facts_in_context": len(facts) - len(missing),
        "facts_in_context_pct": round(100 * (len(facts) - len(missing)) / len(facts), 1) if facts else None,
        "facts_in_gold_chunks": in_gold_chunks,
        "fact_coverage_mean": round(sum(coverages) / len(coverages), 3) if coverages else None,
        "fact_coverage_min": round(min(coverages), 3) if coverages else None,
        "all_facts_in_context": not missing,
        "missing_facts": " | ".join(missing),
        # 3. distractors
        "own_distractor_in_context": any(d in own_distractors for d in context_docs),
        "own_distractor_chunks": sum(d in own_distractors for d in context_docs),
        "other_distractor_chunks": sum(selection.get(d) == "distractor" and d not in own_distractors
                                       for d in context_docs),
        "noise_chunks": sum(selection.get(d) == "noise" and d not in gold for d in context_docs),
        "first_gold_rank": first_rank(lambda d: d in gold),
        "first_own_distractor_rank": first_rank(lambda d: d in own_distractors),
        # 4. repeats
        "chunks_retrieved": n_retrieved,
        "chunks_in_context": len(kept),
        "chunks_dropped_by_budget": n_retrieved - len(kept),
        "distinct_docs_in_context": len(per_doc),
        "repeat_chunks": len(kept) - len(per_doc),
        "max_chunks_one_doc": max(per_doc.values()) if per_doc else 0,
        "gold_chunks": sum(d in gold for d in context_docs),
        "n_gold_docs": len(gold),
        # text, for reading the row
        "correctness_rationale": ev.get("correctness_rationale"),
        "question": q["question"],
        "gold_answer": q["gold_answer"],
        "answer": answer,
    }


def print_summary(label: str, g: pd.DataFrame, *, replayed: bool) -> None:
    """Print the headline checks for one group of rows (wrong or correct answers)."""
    n = len(g)
    print(f"\n=== {n} {label} answers with recall@10 = 1 (Phase 1 verdict) ===")
    if n == 0:
        return
    if replayed:
        print(f"replayed context matches original run: {g['replay_matches'].sum()}/{n}")
    print(f"says 'don't know' or similar:           {g['says_dont_know'].sum()} ({100 * g['says_dont_know'].mean():.0f}%)")
    print(f"all gold facts in prompt context:       {g['all_facts_in_context'].sum()} "
          f"({100 * g['all_facts_in_context'].mean():.0f}%), mean {g['facts_in_context_pct'].mean():.0f}% of facts")
    print(f"own distractor in prompt context:       {g['own_distractor_in_context'].sum()} "
          f"({100 * g['own_distractor_in_context'].mean():.0f}%)")
    print(f"repeat passages per prompt:             mean {g['repeat_chunks'].mean():.1f} of "
          f"{g['chunks_in_context'].mean():.1f}; max from one doc mean {g['max_chunks_one_doc'].mean():.1f}")
    print(f"passages dropped by token budget:       {int(g['chunks_dropped_by_budget'].sum())} total")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyse wrong answers whose gold docs were all retrieved.")
    fixed = settings.paths.results_dir / "phase_1" / "fixed"
    p.add_argument("--answers", type=Path, default=fixed / "answers.jsonl")
    p.add_argument("--eval", type=Path, default=fixed / "cascade" / "answers_eval.jsonl")
    p.add_argument("--include-correct", action="store_true", help="Also include recall=1 rows that were correct.")
    p.add_argument("--contexts", type=Path, default=None,
                   help="Check the passages in this contexts JSONL instead of replaying Phase 1 "
                        "(e.g. from query_small_to_big.py --contexts-only).")
    p.add_argument("--output-dir", type=Path, default=settings.paths.results_dir / "analysis")
    p.add_argument("--output", type=Path, default=None,
                   help="CSV path (default: <output-dir>/p1_recall1_wrong.csv or p1_recall1_all.csv).")
    args = p.parse_args(argv)
    setup_logging()

    qa = load_qa()
    answers = {a["question_id"]: a for a in load_answers(args.answers)}
    evals = [json.loads(line) for line in args.eval.open(encoding="utf-8")]
    targets = [e for e in evals if e.get("recall_at_k") == 1.0 and (args.include_correct or e.get("correct") is False)]
    logger.info("%d questions with recall@10 = 1%s.", len(targets), "" if args.include_correct else " and a wrong answer")

    selection, distractors = load_doc_tags()
    contexts = load_contexts(args.contexts) if args.contexts else None
    pipeline = None if contexts else NaivePipeline()
    rows = []
    for ev in targets:
        qid = ev["question_id"]
        if contexts is not None:
            ctx = contexts[qid]
            texts = [p["text"] for p in ctx["passages"]]
            # `candidates` = passages the pipeline wanted to send before its token budget.
            passages = (texts, [p["doc_id"] for p in ctx["passages"]], ctx.get("candidates", len(texts)))
            replay = None
        else:
            kept, docs, retrieved = phase1_passages(qa.loc[qid, "question"], pipeline)
            passages, replay = (kept, docs, TOP_K), retrieved == answers[qid]["doc_ids"]
        rows.append(analyse(answers[qid], ev, qa.loc[qid], passages, selection, distractors.get(qid, set()), replay))
    df = pd.DataFrame(rows)
    df.insert(2, "correct", [e.get("correct") for e in targets])

    out = args.output or args.output_dir / (
        "phase1_recall1_all.csv" if args.include_correct else "phase1_recall1_wrong.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    groups = [("wrong", df[df["correct"] == False])]  # noqa: E712
    if args.include_correct:
        groups.append(("correct", df[df["correct"] == True]))  # noqa: E712
    for label, g in groups:
        print_summary(label, g, replayed=contexts is None)
    wrong = groups[0][1]
    print("\nwrong answers by question type:")
    print(wrong.groupby("question_type").agg(
        n=("question_id", "size"),
        dont_know=("says_dont_know", "sum"),
        all_facts=("all_facts_in_context", "sum"),
        facts_pct=("facts_in_context_pct", "mean"),
        own_distractor=("own_distractor_in_context", "sum"),
        repeats=("repeat_chunks", "mean"),
    ).round(1).to_string())
    logger.info("Wrote %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
