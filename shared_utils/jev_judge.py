"""Jev (TypeSafe System One) as a cheap first-pass judge, with uncertain verdicts escalated to sol.

Why (cost):
    A full 500-question evaluation with gpt-5.6-sol costs ~$4, >10x the cost of
    answering. Jev bills ~$0.042 per 1M input tokens and nothing for output, so a
    full Jev pass costs ~$0.03. `scripts/compare_jev_judge.py` measured Jev against
    485 cached sol verdicts (easy-corpus Phase 1):
        * Jev alone agrees with sol on ~89-93% of verdicts, but is too lenient on
          long answers with a small conflicting detail (a weekday, a version range).
        * Jev first, with *uncertain* verdicts re-asked to sol ("cascade"), agrees
          with sol on ~98.6% of correctness verdicts and ~98.1% of facts, while
          sending only ~28% of correctness and ~30% of completeness calls to sol:
          ~$1.15 per full evaluation instead of ~$4.
    The cascade keeps sol as the reference judge, so scores stay comparable with
    the sol-only numbers already in the README.

How Jev judges (atomic questions, logic in code -- per the Jev docs):
    Correctness call, state = {question, gold_answer, answer}:
        refusal      Choice  answers / declines
        main_point   Noul    does `answer` state the gold answer's main conclusion?
        contradicts  Noul    does `answer` conflict with `gold_answer`?
        -> info_not_found: correct iff it declines
        -> otherwise:      correct iff it answers, main_point >= MAIN_POINT_MIN and
                           contradicts < CONTRADICTS_MAX
    Completeness call, state = {question, answer} (no gold answer: every fact is
    written in it, so Jev could find the fact there instead of in the answer):
        fact_i       Noul    does `answer` state or clearly imply fact i?

When a verdict counts as uncertain (and goes to sol in cascade mode):
    * the refusal Choice's confidence is below REFUSAL_CONFIDENCE_MIN, or
    * main_point / contradicts falls inside its grey band (non-info_not_found), or
    * for completeness, any fact probability falls inside FACT_BAND; the whole
      question is then re-judged by sol's batched completeness call.
    All thresholds and bands were tuned on the 485 easy-corpus questions; see the
    script's output for the sweep. Re-check them when the corpus changes.

Every Jev response is cached in the shared SQLite cache, so re-scoring is free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from shared_utils.cache import get_cache, make_key
from shared_utils.config import settings
from shared_utils.llm import count_tokens, usage

logger = logging.getLogger(__name__)

# Bump when any question wording below changes, so cached Jev answers produced
# by the old wording are not reused.
PROMPT_VERSION = "jev-judge-v1"
NOT_FOUND_TYPE = "info_not_found"

# Decision thresholds (applied when Jev's verdict is kept).
MAIN_POINT_MIN = 0.7
CONTRADICTS_MAX = 0.3
FACT_MIN = 0.5

# Escalation rules: [low, high) grey bands around each threshold, plus a
# confidence floor for the refusal Choice.
MAIN_POINT_BAND = (0.6, 0.85)
CONTRADICTS_BAND = (0.2, 0.5)
FACT_BAND = (0.3, 0.7)
REFUSAL_CONFIDENCE_MIN = 0.8


# =============================================================================
# Questions
# =============================================================================
# Jev reads literally, so every instruction names the state field it refers to,
# and the criteria spell out boundary cases (paraphrase counts; vaguer does not).
def correctness_questions() -> dict[str, dict[str, Any]]:
    """The three atomic questions combined in code into one correctness verdict."""
    return {
        "refusal": {
            "type": "choice",
            "instructions": "Does `answer` give an answer to `question`, or does it decline?",
            "criteria": {
                "answers": "`answer` provides information that addresses `question`, even if partial or hedged",
                "declines": "`answer` says the information is unavailable, unknown, or not in the provided context",
            },
        },
        "main_point": {
            "type": "noul",
            "instructions": "Does `answer` state the same main conclusion as `gold_answer`?",
            "criteria": {
                "true": "The central claim of `gold_answer` appears in `answer`, possibly reworded or with extra detail",
                "false": "`answer` omits, changes, or only vaguely gestures at the central claim of `gold_answer`",
            },
        },
        "contradicts": {
            "type": "noul",
            "instructions": "Does `answer` contain a statement that conflicts with `gold_answer`?",
            "criteria": {
                "true": "`answer` states a different number, quantity, name, date, version, or fact than `gold_answer`",
                "false": "Everything `answer` says that overlaps with `gold_answer` agrees with it",
            },
        },
    }


def fact_questions(facts: list[str]) -> dict[str, dict[str, Any]]:
    """One independent yes/no question per gold fact."""
    return {
        f"fact_{i}": {
            "type": "noul",
            "instructions": f"Does `answer` state or clearly imply this fact: {fact}",
            "criteria": {
                "true": "`answer` states the fact, or a paraphrase with the same meaning",
                "false": "The fact is missing from `answer`, stated more vaguely, or contradicted",
            },
        }
        for i, fact in enumerate(facts)
    }


# =============================================================================
# Cached client
# =============================================================================
class CachedJev:
    """Cached wrapper around `AsyncTypeSafeClient.system_one`.

    Usage is reported into the shared `usage` tracker (input tokens only: Jev
    output is free), so Jev shows up in the same cost reports as OpenAI models.

    Args:
        model:   Jev model id. Defaults to `settings.jev_model`, a pinned version:
                 an alias like `jev-latest` can move to a new model while the
                 cache keeps serving answers from the old one.
        dry_run: Count tokens and return None instead of calling the API.
    """

    def __init__(self, model: str | None = None, *, dry_run: bool = False) -> None:
        self.model = model or settings.jev_model
        self.dry_run = dry_run
        self.cache = get_cache()
        self._client = None
        self.calls = self.cache_hits = self.input_tokens = self.estimated_tokens = 0

    def _get_client(self):
        if self._client is None:
            # Lazy import + creation: dry runs and fully cached runs never need it.
            from typesafe_sdk import AsyncTypeSafeClient

            self._client = AsyncTypeSafeClient(model=self.model, timeout=120.0)
        return self._client

    def peek(self, state: dict[str, Any], questions: dict[str, dict[str, Any]],
             *, prompt_version: str | None = None) -> dict[str, Any] | None:
        """The cached response for this call, or None. Does not count as a hit."""
        key = make_key("jev", self.model, prompt_version or PROMPT_VERSION, state, questions)
        return self.cache.get_json(key)

    async def ask(self, state: dict[str, Any], questions: dict[str, dict[str, Any]],
                  *, prompt_version: str | None = None) -> dict[str, Any] | None:
        """Return {"answers": {name: ...}, "model": str, "input_tokens": int}, or None in dry-run.

        Noul answers are stored as {"noul": p}; Choice answers as
        {"choice", "probabilities", "confidence"}. `prompt_version` defaults to
        the judge's version; other callers (the relevance filter) pass their own
        so a change to the judge wording does not invalidate their cache.
        """
        key = make_key("jev", self.model, prompt_version or PROMPT_VERSION, state, questions)
        cached = self.cache.get_json(key)
        if cached is not None:
            self.cache_hits += 1
            usage.record_cache_hit(self.model)
            return cached
        if self.dry_run:
            tokens = count_tokens(json.dumps(state) + json.dumps(questions))
            self.estimated_tokens += tokens
            usage.record_estimate(self.model, tokens, 0)
            return None

        response = await self._get_client().system_one(state=state, questions=questions)
        answers: dict[str, Any] = {name: {"noul": float(a.noul)} for name, a in response.nouls.items()}
        for name, a in response.choices.items():
            answers[name] = {"choice": a.choice, "probabilities": dict(a.probabilities),
                             "confidence": float(a.confidence)}
        result = {"answers": answers, "model": response.model, "input_tokens": response.usage.input_tokens or 0}
        self.calls += 1
        self.input_tokens += result["input_tokens"]
        usage.record_call(self.model, result["input_tokens"], 0)
        self.cache.set_json(key, result)
        return result

    async def aclose(self) -> None:
        """Close the HTTP client. Call from the same event loop that used it."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Turning probabilities into verdicts
# =============================================================================
@dataclass(frozen=True)
class CorrectnessSignals:
    """Jev's raw answers to the three correctness questions."""

    refusal_choice: str
    refusal_confidence: float
    p_declines: float
    p_main_point: float
    p_contradicts: float

    @classmethod
    def from_answers(cls, answers: dict[str, Any]) -> CorrectnessSignals:
        refusal = answers["refusal"]
        return cls(
            refusal_choice=refusal["choice"],
            refusal_confidence=refusal["confidence"],
            p_declines=refusal["probabilities"].get("declines", 0.0),
            p_main_point=answers["main_point"]["noul"],
            p_contradicts=answers["contradicts"]["noul"],
        )

    def describe(self) -> str:
        """Compact rationale string, stored where sol would store its sentence."""
        return (f"jev: refusal={self.refusal_choice}({self.refusal_confidence:.2f}) "
                f"main_point={self.p_main_point:.2f} contradicts={self.p_contradicts:.2f}")


def _in_band(p: float, band: tuple[float, float]) -> bool:
    return band[0] <= p < band[1]


def is_correct(sig: CorrectnessSignals, question_type: str,
               main_min: float = MAIN_POINT_MIN, contra_max: float = CONTRADICTS_MAX) -> bool:
    """Combine the atomic answers into one correctness verdict."""
    if question_type == NOT_FOUND_TYPE:
        return sig.refusal_choice == "declines"
    return sig.refusal_choice == "answers" and sig.p_main_point >= main_min and sig.p_contradicts < contra_max


def correctness_uncertain(sig: CorrectnessSignals, question_type: str) -> bool:
    """True when the cascade should hand this correctness verdict to sol."""
    if sig.refusal_confidence < REFUSAL_CONFIDENCE_MIN:
        return True
    if question_type == NOT_FOUND_TYPE:
        return False
    return _in_band(sig.p_main_point, MAIN_POINT_BAND) or _in_band(sig.p_contradicts, CONTRADICTS_BAND)


def fact_probabilities(answers: dict[str, Any], n_facts: int) -> list[float]:
    """Per-fact Noul probabilities, in fact order."""
    return [answers[f"fact_{i}"]["noul"] for i in range(n_facts)]


def facts_uncertain(p_facts: list[float]) -> bool:
    """True when any fact is borderline, so sol should re-judge the question's completeness."""
    return any(_in_band(p, FACT_BAND) for p in p_facts)


# =============================================================================
# Judge facade used by shared_utils.evaluation
# =============================================================================
class JevJudge:
    """Ask Jev for one question's correctness signals or fact probabilities.

    Returns None from either method in dry-run mode (nothing was asked).
    """

    def __init__(self, *, dry_run: bool = False) -> None:
        self.jev = CachedJev(dry_run=dry_run)

    async def correctness(self, question: str, gold_answer: str, answer: str) -> CorrectnessSignals | None:
        res = await self.jev.ask({"question": question, "gold_answer": gold_answer, "answer": answer},
                                 correctness_questions())
        return None if res is None else CorrectnessSignals.from_answers(res["answers"])

    async def fact_probs(self, question: str, answer: str, facts: list[str]) -> list[float] | None:
        res = await self.jev.ask({"question": question, "answer": answer}, fact_questions(facts))
        return None if res is None else fact_probabilities(res["answers"], len(facts))

    async def aclose(self) -> None:
        await self.jev.aclose()
