"""Local re-implementation of the EnterpriseRAG-Bench leaderboard metrics.

Every phase writes its answers to a JSONL file with one object per question:

    {"question_id": "qst_0001", "answer": "The default limits are ...", "doc_ids": ["dsid_...", ...]}

    * `answer`  - the system's final natural-language answer.
    * `doc_ids` - the documents the system retrieved/used, best first. Only the
                  ranking matters, duplicates are ignored.

This module scores that file exactly the way the benchmark paper defines the
metrics (arXiv 2605.05253, Section 5):

    Correctness (0/1)     LLM judge: is the answer broadly aligned with the gold
                          answer, with no conflicting facts or mismatched numbers?
                          Lenient on style and extra detail. The judge does NOT
                          see the answer facts (kept independent, as in the paper).
    Completeness (0-1)    Fraction of the gold `answer_facts` the answer contains
                          or implies. The judge does NOT see the correctness verdict.
    Document Recall@k     |top-k retrieved ∩ gold| / |gold|. Only for question
                          types that have gold docs (not High Level / Info Not Found).
    Invalid Extra Docs    Number of retrieved docs not in the gold set, again only
                          for types with gold docs. Lower is better.
    Overall Score         mean(correct × completeness): the leaderboard's primary
                          metric. A wrong answer scores 0 however complete it is.

"Info Not Found" questions need no special casing: their gold answer states that
the answer must say the information is unavailable, so the correctness judge
handles them naturally.

Two honest differences from the official harness:
    * The official Invalid Extra Docs excludes docs a 3-judge panel marks "valid"
      (relevant but not required). We do not run that (costly) panel, so our
      number is a strict upper bound: it counts every non-gold doc.
    * The official harness uses GPT-5.4 as judge; our reference judge is
      `settings.judge_model`. Scores are comparable between our phases, and close
      to (not identical to) leaderboard numbers.

Judge modes (--judge, default `settings.judge_mode`; see shared_utils/jev_judge.py):
    cascade   Jev judges first; only verdicts Jev is uncertain about are re-asked
              to `judge_model`. ~98% agreement with `sol` at ~30% of its cost.
    sol       `judge_model` on everything: the reference, ~$4 per 500 questions.
    jev       Jev only: ~$0.03 per 500 questions, ~90% agreement with `sol`.
    Each row of the per-question output records which judge decided it.

Cost controls:
    --no-llm          Retrieval metrics only (recall, invalid extra docs): $0.
    --dry-run         Price the judge calls without making them. In cascade mode
                      this assumes every verdict escalates (an upper bound).
    --limit N         Score only the first N questions.
    Batched facts     By default the facts of one question are judged in ONE call
                      per group of up to 12 facts (FACT_GROUP_SIZE) that returns a
                      verdict per fact (~N× fewer calls). Use --isolated-facts for
                      the paper's one-call-per-fact protocol.
    Cache             Every verdict is cached, so re-scoring unchanged answers is free.
    Empty answers     Scored 0 without calling the judge.

Usage:
    python -m shared_utils.evaluation results/phase_1/fixed/answers.jsonl --no-llm
    python -m shared_utils.evaluation results/phase_1/fixed/answers.jsonl --dry-run
    python -m shared_utils.evaluation results/phase_1/fixed/answers.jsonl --limit 50

Reads:  the answers JSONL, data/raw_onyx_subset/mini_redwood_qa.jsonl.
Writes: <output-dir>/<answers-stem>_eval.jsonl   (one row per question)
        <output-dir>/<answers-stem>_metrics.json (overall + per question type + cost)
        results/log.jsonl                        (one line per completed run; see results/README.md)

Phase scripts pass an output dir of results/<phase>/<run>/<judge>/, so a run
judged twice (e.g. jev, then cascade) keeps both verdicts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from shared_utils import jev_judge
from shared_utils.config import settings, setup_logging
from shared_utils.llm import CachedChat, run_limited, usage

logger = logging.getLogger(__name__)

JUDGE_MODES = ("cascade", "sol", "jev")

# Bump these whenever the corresponding prompt text below changes, so cached
# verdicts produced by the old wording are not reused.
CORRECTNESS_PROMPT_VERSION = "correctness-v1"
COMPLETENESS_PROMPT_VERSION = "completeness-v1"

# Max facts per batched completeness call. With 40+ facts and a long answer, the
# judge spent all of `judge_max_tokens` reasoning and returned nothing; failed
# replies are not cached, so every re-evaluation paid for them again and then
# fell back to one call per fact. Groups of 12 fit comfortably. Questions with
# <= 12 facts send exactly the same single request as before (cache stays valid).
FACT_GROUP_SIZE = 12


# =============================================================================
# Judge prompts & structured outputs
# =============================================================================
# The rules below paraphrase the benchmark's own definitions (paper, Sec. 5.2).
CORRECTNESS_SYSTEM = """You are grading an answer produced by a company-internal search assistant.

Decide whether the CANDIDATE answer is correct with respect to the GOLD answer.

Mark it correct if it is broadly aligned with the gold answer:
- Be lenient about wording, style, ordering, and extra context or supplementary detail.
- It must NOT contain facts that conflict with the gold answer.
- Numbers, quantities, names, dates and versions must not be mismatched.
- If the gold answer says the information is not available or the answer must be caveated,
  the candidate is correct only if it clearly says the information is not (fully) available.

Keep the rationale to one short sentence."""

CORRECTNESS_USER = """QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

CANDIDATE ANSWER:
{answer}"""


class CorrectnessVerdict(BaseModel):
    """Structured output of the correctness judge."""

    rationale: str = Field(description="One short sentence explaining the verdict.")
    correct: bool = Field(description="True if the candidate is broadly aligned with the gold answer.")


COMPLETENESS_SYSTEM = """You are checking which facts are supported by an answer from a company-internal search assistant.

For EACH numbered fact, decide independently whether the CANDIDATE answer states or clearly implies it.
- Judge each fact on its own; do not let one fact's verdict influence another.
- Paraphrases count. Missing, vaguer, or contradicted facts do not.
Return exactly one verdict per fact, using the fact's number as `index`."""

COMPLETENESS_USER = """QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

FACTS:
{facts}"""


class FactVerdict(BaseModel):
    """Verdict for a single fact."""

    index: int = Field(description="The fact's number from the list.")
    supported: bool = Field(description="True if the answer states or clearly implies this fact.")


class FactVerdicts(BaseModel):
    """Structured output of the batched completeness judge."""

    verdicts: list[FactVerdict]


class SingleFactVerdict(BaseModel):
    """Structured output of the isolated (one-fact) completeness judge."""

    supported: bool = Field(description="True if the answer states or clearly implies the fact.")


# =============================================================================
# Helpers
# =============================================================================
# Citations are stripped before judging (as the official harness does) so the
# judge scores substance, not reference formatting. Patterns cover the styles
# our phases emit: [dsid_...], (dsid_...), bare dsid_ ids, [1], [1, 2], [doc 3].
_CITATION_PATTERNS = [
    re.compile(r"[\[(]\s*(?:source|doc|document)?s?:?\s*dsid_[0-9a-f]{8,}(?:\s*[,;]\s*dsid_[0-9a-f]{8,})*\s*[\])]", re.I),
    re.compile(r"\bdsid_[0-9a-f]{8,}\b"),
    re.compile(r"\[(?:doc(?:ument)?\s*)?\d+(?:\s*[,;-]\s*\d+)*\]", re.I),
]


def strip_citations(text: str) -> str:
    """Remove citation markers and tidy leftover whitespace."""
    for pat in _CITATION_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)  # "foo [1]." became "foo ." -> "foo."
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def dedupe(ids: list[str]) -> list[str]:
    """Drop repeated doc IDs, keeping the first (best-ranked) occurrence.

    Needed because chunk-level retrievers often return several chunks of the
    same document; the benchmark scores documents, not chunks.
    """
    seen: set[str] = set()
    return [d for d in ids if not (d in seen or seen.add(d))]


def load_qa(path: Path | None = None) -> pd.DataFrame:
    """Load the curated QA set (defaults to mini_redwood_qa.jsonl), indexed by question_id."""
    path = path or settings.paths.mini_qa
    if not path.is_file():
        raise FileNotFoundError(f"QA file not found: {path}. Run scripts/curate_mini_redwood.py first.")
    return pd.read_json(path, lines=True).set_index("question_id", drop=False)


def load_answers(path: Path) -> list[dict[str, Any]]:
    """Read a phase's answers JSONL and validate each row's shape."""
    if not path.is_file():
        raise FileNotFoundError(f"Answers file not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "question_id" not in row:
                raise ValueError(f"{path}:{n} is missing 'question_id'")
            row.setdefault("answer", "")
            row["doc_ids"] = [str(d) for d in (row.get("doc_ids") or [])]
            rows.append(row)
    return rows


def save_answers(rows: list[dict[str, Any]], path: Path) -> None:
    """Write answers in the format this module expects (used by every phase)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"question_id": r["question_id"], "answer": r.get("answer", ""),
                                "doc_ids": r.get("doc_ids", [])}, ensure_ascii=False) + "\n")


# =============================================================================
# Per-question scoring
# =============================================================================
@dataclass
class QuestionScore:
    """All metrics for one question. `None` means "not applicable / not judged"."""

    question_id: str
    question_type: str
    n_gold: int
    n_retrieved: int
    recall_at_k: float | None
    invalid_extra_docs: int | None
    correct: bool | None
    completeness: float | None
    score: float | None
    correctness_rationale: str | None = None
    unsupported_facts: list[str] | None = None
    # Which judge decided each metric: "jev", "sol", or "none" (no call needed).
    correctness_judge: str | None = None
    completeness_judge: str | None = None
    # Words in the answer as judged (citations stripped). Tracked because longer
    # answers cost more to generate and to judge, and can inflate completeness.
    answer_words: int = 0


def retrieval_metrics(doc_ids: list[str], gold: list[str], k: int) -> tuple[float | None, int | None]:
    """Recall@k and invalid-extra-doc count for one question (free, no LLM).

    Returns (None, None) for question types without gold documents, matching
    the benchmark, which excludes them from both metrics.
    """
    if not gold:
        return None, None
    gold_set = set(gold)
    ranked = dedupe(doc_ids)
    recall = len(set(ranked[:k]) & gold_set) / len(gold_set)
    invalid = sum(1 for d in ranked if d not in gold_set)
    return recall, invalid


class Judge:
    """Judge for correctness and completeness: Jev, the LLM judge, or a cascade of both.

    Args:
        mode:           "cascade", "sol" or "jev" (default `settings.judge_mode`).
        dry_run:        Price calls instead of making them.
        isolated_facts: For the LLM judge, one call per fact (paper protocol)
                        instead of one batched call per question (cheaper; default).
    """

    def __init__(self, *, mode: str | None = None, dry_run: bool = False, isolated_facts: bool = False) -> None:
        self.mode = mode or settings.judge_mode
        if self.mode not in JUDGE_MODES:
            raise ValueError(f"Unknown judge mode {self.mode!r}; expected one of {JUDGE_MODES}")
        self.llm = CachedChat(settings.judge_model, max_tokens=settings.judge_max_tokens,
                              reasoning_effort=settings.judge_reasoning_effort, dry_run=dry_run)
        self.jev = jev_judge.JevJudge(dry_run=dry_run) if self.mode != "sol" else None
        self.isolated_facts = isolated_facts

    async def aclose(self) -> None:
        if self.jev is not None:
            await self.jev.aclose()

    async def correctness(self, question: str, gold_answer: str, answer: str,
                          question_type: str) -> tuple[bool | None, str | None, str]:
        """Return (correct, rationale, judge); correct is None in dry-run mode.

        In cascade mode Jev answers first; its verdict is kept unless it is
        uncertain, in which case the LLM judge decides. A dry run has no Jev
        answer to inspect, so it prices the LLM call too (upper bound).
        """
        if self.jev is not None:
            sig = await self.jev.correctness(question, gold_answer, answer)
            if sig is not None and (self.mode == "jev" or not jev_judge.correctness_uncertain(sig, question_type)):
                return jev_judge.is_correct(sig, question_type), sig.describe(), "jev"
            if self.mode == "jev":
                return None, None, "jev"
        correct, rationale = await self._llm_correctness(question, gold_answer, answer)
        return correct, rationale, "sol"

    async def completeness(self, question: str, answer: str,
                           facts: list[str]) -> tuple[float | None, list[str], str]:
        """Return (fraction of facts supported, unsupported facts, judge).

        In cascade mode, if any fact's Jev probability is borderline, the whole
        question is re-judged by the LLM judge (one batched call), so a question's
        completeness always comes from a single judge.
        """
        if not facts:
            return 1.0, [], "none"
        if self.jev is not None:
            probs = await self.jev.fact_probs(question, answer, facts)
            if probs is not None and (self.mode == "jev" or not jev_judge.facts_uncertain(probs)):
                ok = [p >= jev_judge.FACT_MIN for p in probs]
                return sum(ok) / len(facts), [f for f, good in zip(facts, ok) if not good], "jev"
            if self.mode == "jev":
                return None, [], "jev"
        completeness, unsupported = await self._llm_completeness(question, answer, facts)
        return completeness, unsupported, "sol"

    async def _llm_correctness(self, question: str, gold_answer: str, answer: str) -> tuple[bool | None, str | None]:
        """Return (correct, rationale); (None, None) in dry-run mode."""
        res = await self.llm.ainvoke(
            [("system", CORRECTNESS_SYSTEM),
             ("user", CORRECTNESS_USER.format(question=question, gold_answer=gold_answer, answer=answer))],
            prompt_version=CORRECTNESS_PROMPT_VERSION,
            schema=CorrectnessVerdict,
        )
        if res.parsed is None:
            return None, None
        return bool(res.parsed["correct"]), res.parsed["rationale"]

    async def _fact_isolated(self, question: str, answer: str, fact: str) -> bool | None:
        res = await self.llm.ainvoke(
            [("system", COMPLETENESS_SYSTEM),
             ("user", COMPLETENESS_USER.format(question=question, answer=answer, facts=f"1. {fact}"))],
            prompt_version=COMPLETENESS_PROMPT_VERSION + "-isolated",
            schema=SingleFactVerdict,
        )
        return None if res.parsed is None else bool(res.parsed["supported"])

    async def _fact_group(self, question: str, answer: str, group: list[str]) -> dict[int, bool] | None:
        """Judge one group of facts in one call.

        Facts are numbered 1..len(group) within the group, so a group of <= 12
        facts is byte-identical to the original one-call-per-question request.

        Returns:
            {index within group: supported} for the facts the reply covered
            (may be partial or empty), or None in dry-run mode.
        """
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(group, 1))
        res = await self.llm.ainvoke(
            [("system", COMPLETENESS_SYSTEM),
             ("user", COMPLETENESS_USER.format(question=question, answer=answer, facts=numbered))],
            prompt_version=COMPLETENESS_PROMPT_VERSION,
            schema=FactVerdicts,
        )
        if res.dry_run:
            return None
        return {v["index"]: bool(v["supported"])
                for v in (res.parsed or {}).get("verdicts", []) if 1 <= v["index"] <= len(group)}

    async def _llm_completeness(self, question: str, answer: str, facts: list[str]) -> tuple[float | None, list[str]]:
        """Return (fraction of facts supported, list of unsupported facts).

        Batched mode sends the facts in groups of `FACT_GROUP_SIZE`, one call per
        group, run concurrently. Facts a group's reply does not cover are re-asked
        one by one, so a partial reply never forces paying for the group again.
        """
        if not facts:
            return 1.0, []

        verdicts: dict[int, bool | None] = {}  # 1-based index into `facts`
        if not self.isolated_facts:
            starts = range(0, len(facts), FACT_GROUP_SIZE)
            replies = await asyncio.gather(
                *(self._fact_group(question, answer, facts[s:s + FACT_GROUP_SIZE]) for s in starts))
            if any(r is None for r in replies):
                return None, []
            for s, reply in zip(starts, replies):
                verdicts.update({s + i: ok for i, ok in reply.items()})

        missing = [i for i in range(1, len(facts) + 1) if i not in verdicts]
        if missing and not self.isolated_facts:
            logger.debug("Batched judge skipped %d facts; re-asking them individually.", len(missing))
        results = await asyncio.gather(*(self._fact_isolated(question, answer, facts[i - 1]) for i in missing))
        verdicts.update(zip(missing, results))

        if any(v is None for v in verdicts.values()):
            return None, []  # dry run or unparseable verdicts
        supported = sum(verdicts.values())
        unsupported = [facts[i - 1] for i, ok in sorted(verdicts.items()) if not ok]
        return supported / len(facts), unsupported


async def score_question(row: dict[str, Any], qa: pd.Series, judge: Judge | None, k: int) -> QuestionScore:
    """Compute every metric for one answered question."""
    gold = list(qa["expected_doc_ids"])
    recall, invalid = retrieval_metrics(row["doc_ids"], gold, k)
    answer = strip_citations(row.get("answer") or "")

    correct = completeness = score = None
    rationale: str | None = None
    unsupported: list[str] | None = None
    corr_judge = comp_judge = None
    if judge is not None:
        if not answer:
            # No answer -> nothing to judge. Scoring it 0 directly saves 2 calls.
            correct, completeness, score, rationale = False, 0.0, 0.0, "empty answer"
            unsupported = list(qa["answer_facts"])
            corr_judge = comp_judge = "none"
        else:
            # The two judgments are independent, so run them concurrently.
            (correct, rationale, corr_judge), (completeness, unsupported, comp_judge) = await asyncio.gather(
                judge.correctness(qa["question"], qa["gold_answer"], answer, qa["question_type"]),
                judge.completeness(qa["question"], answer, list(qa["answer_facts"])),
            )
            if correct is not None and completeness is not None:
                score = float(correct) * completeness

    return QuestionScore(
        question_id=row["question_id"],
        question_type=qa["question_type"],
        n_gold=len(gold),
        n_retrieved=len(dedupe(row["doc_ids"])),
        recall_at_k=recall,
        invalid_extra_docs=invalid,
        correct=correct,
        completeness=completeness,
        score=score,
        correctness_rationale=rationale,
        unsupported_facts=unsupported,
        correctness_judge=corr_judge,
        completeness_judge=comp_judge,
        answer_words=len(answer.split()),
    )


# =============================================================================
# Aggregation
# =============================================================================
def aggregate(scores: list[QuestionScore]) -> dict[str, Any]:
    """Overall and per-question-type averages.

    Each metric is averaged only over the questions where it applies (e.g.
    recall ignores High Level / Info Not Found), mirroring the leaderboard.
    Percent metrics are reported on a 0-100 scale; invalid extra docs is a count.
    """
    df = pd.DataFrame([asdict(s) for s in scores])
    if df.empty:
        return {"overall": {}, "by_type": {}}

    def summarise(g: pd.DataFrame) -> dict[str, Any]:
        def pct(col: str) -> float | None:
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            return None if s.empty else round(100 * s.mean(), 2)

        inv = pd.to_numeric(g["invalid_extra_docs"], errors="coerce").dropna()
        return {
            "n": int(len(g)),
            "overall_score": pct("score"),
            "correctness": pct("correct"),
            "completeness": pct("completeness"),
            "recall_at_k": pct("recall_at_k"),
            "invalid_extra_docs": None if inv.empty else round(float(inv.mean()), 2),
            "avg_answer_words": round(float(g["answer_words"].mean()), 1),
        }

    return {
        "overall": summarise(df),
        "by_type": {t: summarise(g) for t, g in df.groupby("question_type", sort=True)},
    }


def format_table(metrics: dict[str, Any]) -> str:
    """Render the aggregate metrics as a fixed-width text table."""
    cols = ["n", "overall_score", "correctness", "completeness", "recall_at_k", "invalid_extra_docs",
            "avg_answer_words"]
    rows = [{"type": "OVERALL", **metrics["overall"]}] + [
        {"type": t, **m} for t, m in metrics["by_type"].items()
    ]
    # Unjudged metrics are None; coercing to numeric turns them into NaN so they
    # print as "-" instead of "None".
    df = pd.DataFrame(rows).set_index("type").reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    df["n"] = df["n"].astype(int)
    return df.to_string(na_rep="-", float_format=lambda x: f"{x:.2f}")


def judge_sources(scores: list[QuestionScore]) -> dict[str, dict[str, int]]:
    """How many correctness / completeness verdicts each judge decided.

    In cascade mode the "sol" count is the number of escalations, i.e. what
    drives the evaluation's cost.
    """
    out: dict[str, dict[str, int]] = {"correctness": {}, "completeness": {}}
    for s in scores:
        for metric, src in (("correctness", s.correctness_judge), ("completeness", s.completeness_judge)):
            if src:
                out[metric][src] = out[metric].get(src, 0) + 1
    return out


def append_run_log(entry: dict[str, Any]) -> None:
    """Append one record to results/log.jsonl, the automatic index of every run.

    results/README.md is the human-readable summary; this log is what keeps it
    completable, since every answering run and evaluation adds a line here
    (dry runs do not). `entry` is merged with a timestamp.
    """
    from datetime import datetime

    path = settings.paths.results_dir / "log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"), **entry}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def judge_cost_usd(*, estimated: bool = False) -> float | None:
    """Dollars spent on the judge models (sol + Jev) so far in this process.

    Counts only judge models, so it stays correct even if the caller also
    answered questions in the same process. None if a judge model has no price.
    """
    models = usage.report()["models"]
    field = "estimated_cost_usd" if estimated else "cost_usd"
    total = 0.0
    for name in (settings.judge_model, settings.jev_model):
        u = models.get(name)
        if u is None or not (u["estimated_calls"] if estimated else u["calls"]):
            continue
        if u[field] is None:
            return None
        total += u[field]
    return round(total, 4)


# =============================================================================
# Entry points
# =============================================================================
async def evaluate(
    answers: list[dict[str, Any]],
    qa: pd.DataFrame,
    *,
    use_llm: bool = True,
    dry_run: bool = False,
    isolated_facts: bool = False,
    k: int | None = None,
    judge_mode: str | None = None,
) -> tuple[list[QuestionScore], dict[str, Any]]:
    """Score a list of answers. Importable so phases can evaluate in-process.

    Args:
        answers:  Rows of {question_id, answer, doc_ids}.
        qa:       QA DataFrame from `load_qa()` (indexed by question_id).
        use_llm:  False = retrieval metrics only (no cost).
        dry_run:  Estimate judge cost instead of calling it.
        isolated_facts: One LLM-judge call per fact instead of one per question.
        k:        Recall cutoff (defaults to `settings.recall_k`, i.e. 10).
        judge_mode: "cascade", "sol" or "jev" (defaults to `settings.judge_mode`).

    Returns:
        (per-question scores, aggregate metrics dict).
    """
    k = k or settings.recall_k
    known = [a for a in answers if a["question_id"] in qa.index]
    unknown = len(answers) - len(known)
    if unknown:
        logger.warning("%d answers reference unknown question_ids and are ignored.", unknown)
    if len(known) < len(qa):
        logger.info("Scoring %d of %d questions.", len(known), len(qa))

    judge = Judge(mode=judge_mode, dry_run=dry_run, isolated_facts=isolated_facts) if use_llm else None
    try:
        scores = await run_limited(
            known,
            lambda a: score_question(a, qa.loc[a["question_id"]], judge, k),
            desc="judging" if judge else "",
        )
    finally:
        if judge is not None:
            await judge.aclose()
    return scores, aggregate(scores)


def selected_question_types(args: argparse.Namespace) -> set[str] | None:
    """Question types named by `--question-types` and/or `--question-type`.

    Both flags may be set; the result is their union. None means every type.
    """
    found: list[str] = []
    raw = getattr(args, "question_types", None)
    if raw:
        found.extend(part.strip() for part in raw.split(",") if part.strip())
    single = getattr(args, "question_type", None)
    if single:
        found.append(single.strip())
    return set(found) or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI flags (see module docstring)."""
    p = argparse.ArgumentParser(description="Score a phase's answers with EnterpriseRAG-Bench metrics.")
    p.add_argument("answers", type=Path, help="Answers JSONL: {question_id, answer, doc_ids}.")
    p.add_argument("--qa", type=Path, default=None, help="QA JSONL (default: mini_redwood_qa.jsonl).")
    p.add_argument("--k", type=int, default=settings.recall_k, help="Recall cutoff (default 10).")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N answers.")
    p.add_argument("--question-type", default=None, help="Score only this question type.")
    p.add_argument("--question-types", default=None,
                   help="Score only these question types, comma-separated. Combines with --question-type.")
    p.add_argument("--no-llm", action="store_true", help="Retrieval metrics only; no API calls.")
    p.add_argument("--dry-run", action="store_true", help="Estimate judge cost without calling the API.")
    p.add_argument("--isolated-facts", action="store_true",
                   help="One LLM-judge call per fact (paper protocol, more calls) instead of one per question.")
    p.add_argument("--judge", choices=JUDGE_MODES, default=settings.judge_mode,
                   help=f"Who judges (default {settings.judge_mode}): Jev first with uncertain verdicts "
                        "escalated to the LLM judge, the LLM judge only, or Jev only.")
    p.add_argument("--output-dir", type=Path, default=settings.paths.results_dir)
    return p.parse_args(argv)


async def evaluate_file(args: argparse.Namespace) -> int:
    """Load, score, print and persist one answers file. Returns an exit code.

    Async so callers that are already inside an event loop (the phase runner,
    which answers and then scores in one run) can `await` it. Using a single
    event loop matters: the OpenAI client keeps a shared connection pool bound
    to the loop it was first used in, so a second `asyncio.run()` in the same
    process fails with "Event loop is closed".
    """
    try:
        qa = load_qa(args.qa)
        answers = load_answers(args.answers)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        return 1
    if args.limit:
        answers = answers[: args.limit]
    types = selected_question_types(args)
    if types:
        qa = qa[qa["question_type"].isin(types)]
        answers = [a for a in answers if a["question_id"] in qa.index]
        if not answers:
            logger.error("No answers left after filtering to %s.", ", ".join(sorted(types)))
            return 1

    use_llm = not args.no_llm
    if use_llm and not args.dry_run:
        # Cached verdicts would still work, but failing fast is clearer than a
        # half-finished run that crashes on the first cache miss.
        if args.judge != "jev" and not settings.openai_api_key:
            logger.error("OPENAI_API_KEY is not set. Use --no-llm, --dry-run or --judge jev, or add the key.")
            return 1
        if args.judge != "sol" and not settings.typesafe_api_key:
            logger.error("TYPESAFE_API_KEY is not set. Use --judge sol, --no-llm or --dry-run, or add the key.")
            return 1

    scores, metrics = await evaluate(answers, qa, use_llm=use_llm, dry_run=args.dry_run,
                                     isolated_facts=args.isolated_facts, k=args.k, judge_mode=args.judge)

    mode = "retrieval-only" if not use_llm else ("dry-run" if args.dry_run else "full")
    sources = judge_sources(scores) if use_llm else None
    judge_usd = judge_cost_usd(estimated=args.dry_run) if use_llm else 0.0
    report = {
        "answers_file": str(args.answers),
        "mode": mode,
        "judge_mode": args.judge if use_llm else None,
        "judge_model": settings.judge_model if use_llm and args.judge != "jev" else None,
        "jev_model": settings.jev_model if use_llm and args.judge != "sol" else None,
        "recall_k": args.k,
        **metrics,
        "judge_sources": sources,
        # Dollars this run paid for judging (dry run: the upper-bound estimate).
        # Cache hits are free, so re-scoring unchanged answers reports ~$0.
        "judge_cost_usd": judge_usd,
        "judge_cost": usage.report(),
    }

    judge_label = f", judge={args.judge}" if use_llm else ""
    print(f"\n=== {args.answers.name} ({mode}{judge_label}, Recall@{args.k}) ===")
    print(format_table(metrics))
    if sources and not args.dry_run:
        print(f"[judge] correctness decided by {sources['correctness']}, "
              f"completeness by {sources['completeness']}")
    if use_llm:
        cost = "unknown (set MODEL_PRICES)" if judge_usd is None else f"${judge_usd:.4f}"
        print(f"[judge] cost this run: {cost}{' (dry-run upper bound)' if args.dry_run else ''}; "
              f"avg answer length {metrics['overall'].get('avg_answer_words', 0):.0f} words")
    for line in usage.summary_lines() if use_llm else []:
        print(f"[cost] {line}")

    if not args.dry_run:
        stem = args.answers.stem
        args.output_dir.mkdir(parents=True, exist_ok=True)
        per_q = args.output_dir / f"{stem}_eval.jsonl"
        with per_q.open("w", encoding="utf-8") as f:
            for s in scores:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
        summary = args.output_dir / f"{stem}_metrics.json"
        summary.write_text(json.dumps(report, indent=2))
        logger.info("Wrote %s and %s", per_q, summary)
        append_run_log({
            "kind": "eval", "answers": str(args.answers), "output_dir": str(args.output_dir),
            "mode": mode, "judge": args.judge if use_llm else None,
            "judge_model": settings.judge_model, "jev_model": settings.jev_model,
            "judge_reasoning_effort": settings.judge_reasoning_effort,
            "overall": metrics.get("overall"), "judge_cost_usd": judge_usd,
        })
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    setup_logging()
    try:
        return asyncio.run(evaluate_file(args))
    except KeyboardInterrupt:
        logger.error("Interrupted. Verdicts obtained so far are cached; re-run to resume for free.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
