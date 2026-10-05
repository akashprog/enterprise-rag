"""Central, typed configuration for the whole Enterprise RAG series.

Why one settings object?
    Every phase (baseline, data mastery, Jev reranker, agent router) must be
    comparable. If chunk sizes, top-k, or model names were scattered across
    scripts, a "better" score could silently come from a different model or a
    bigger k. Keeping every knob here makes each phase's change explicit.

How values are resolved (highest priority first):
    1. Real environment variables  (e.g. `ANSWER_MODEL=... python ...`)
    2. The `.env` file in the repo root (copy `.env.example` to start)
    3. The defaults declared below

Usage:
    from shared_utils.config import settings
    settings.answer_model          # -> "gpt-5.6-luna"
    settings.paths.mini_docs       # -> Path(".../data/raw_onyx_subset/mini_redwood_docs.jsonl")

Reads:  `.env` (optional).
Writes: nothing.
"""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The repo root is the parent of this `shared_utils/` package. Resolving paths
# from here (instead of the current working directory) means scripts behave the
# same whether you run them from the repo root or from inside a phase folder.
REPO_ROOT = Path(__file__).resolve().parent.parent


class ModelPrice(BaseModel):
    """Price of one model in USD per 1M tokens.

    Attributes:
        input:  Cost per 1M prompt (input) tokens.
        output: Cost per 1M completion (output) tokens. Embedding models have
                no output tokens, so this is 0 for them.
    """

    input: float
    output: float = 0.0


class Paths(BaseModel):
    """Every file-system location the series reads or writes.

    Attributes are derived from `REPO_ROOT` so they are absolute and stable.
    """

    root: Path = REPO_ROOT
    # The unzipped `all_documents.zip` from the Onyx GitHub release (~512k .txt
    # files, 3.2 GB). Layout: all_documents/<source_type>/<nested dirs>/dsid_<32 hex>__<slug>.txt
    corpus_dir: Path = REPO_ROOT / "all_documents"
    # The release also ships the 500 benchmark questions next to the documents.
    # Reading them locally is free and works offline; Hugging Face is the fallback.
    local_questions: Path = REPO_ROOT / "all_documents" / "questions.jsonl"
    subset_dir: Path = REPO_ROOT / "data" / "raw_onyx_subset"
    mini_docs: Path = REPO_ROOT / "data" / "raw_onyx_subset" / "mini_redwood_docs.jsonl"
    mini_qa: Path = REPO_ROOT / "data" / "raw_onyx_subset" / "mini_redwood_qa.jsonl"
    # SQLite cache of every paid API response (see shared_utils/cache.py).
    cache_db: Path = REPO_ROOT / ".cache" / "api_cache.sqlite"
    # Per-phase answers, metrics and cost reports.
    results_dir: Path = REPO_ROOT / "results"


class Settings(BaseSettings):
    """All tunable knobs for the series.

    Field names map case-insensitively to environment variables, so
    `answer_model` is overridden by `ANSWER_MODEL` in the environment or `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        # Ignore unrelated variables in .env instead of crashing on them.
        extra="ignore",
    )

    # ------------------------------------------------------------------ keys
    # Kept optional so that free, offline tools (curation, --no-llm evaluation,
    # --dry-run estimates) still work on a machine without any keys.
    openai_api_key: str | None = None
    typesafe_api_key: str | None = None

    # ---------------------------------------------------------- infrastructure
    qdrant_url: str = "http://localhost:6333"

    # ----------------------------------------------------------------- models
    # Cost principle: use the cheapest model that can do each job.
    #   - answering + query expansion happen once per question -> small model.
    #   - judging decides our scores, so it gets the stronger model; it also only
    #     runs during evaluation, never in the production retrieval path.
    answer_model: str = "gpt-5.6-luna"
    expansion_model: str = "gpt-5.6-luna"
    judge_model: str = "gpt-5.6-sol"
    # text-embedding-3-small: ~$0.02 / 1M tokens, good quality/price ratio.
    # Embedding the ~4.7k-doc mini corpus costs on the order of cents.
    embedding_model: str = "text-embedding-3-small"
    # Jev (TypeSafe System One) model id: first-pass judge, and from Step 4 the
    # reranker and router. Pinned rather than `jev-latest`, because an alias can
    # move to a new model while cached answers from the old one keep being served.
    jev_model: str = "jev-1.13.0"
    # Who judges correctness/completeness (see shared_utils/jev_judge.py):
    #   "cascade" - Jev first, uncertain verdicts re-asked to `judge_model`
    #               (~98% agreement with sol-only at ~30% of its cost)
    #   "sol"     - `judge_model` on everything (reference, ~$4 per 500 questions)
    #   "jev"     - Jev only (~$0.03 per 500 questions; ~90% agreement)
    judge_mode: str = "cascade"

    # ---------------------------------------------------------------- pricing
    # USD per 1M tokens, used for dry-run estimates and cost reports. Only the
    # embedding price is pre-filled because it is stable and well known; fill in
    # chat-model prices in `.env` (MODEL_PRICES as JSON). Unknown models are still
    # token-counted -- their dollar cost is just reported as "unknown".
    model_prices: dict[str, ModelPrice] = Field(
        default_factory=lambda: {"text-embedding-3-small": ModelPrice(input=0.02)}
    )

    # ------------------------------------------------------- budgets & limits
    # Max concurrent in-flight API calls. Higher = faster, but too high trips
    # provider rate limits (429s), which waste time on retries.
    max_concurrency: int = 8
    # Retries with exponential backoff for transient errors (429 / 5xx / timeouts).
    max_retries: int = 4
    # Embeddings are billed per token, not per request, so large batches cost
    # the same but make far fewer round-trips (fewer chances to hit rate limits).
    embedding_batch_size: int = 256
    # Hard caps on generated tokens. Output tokens are the most expensive tokens,
    # and a runaway answer adds cost without adding benchmark score. On reasoning
    # models the cap also covers hidden reasoning tokens, so it cannot be too
    # tight or the visible reply gets cut off (llm.py warns when that happens).
    # 4,000 rather than 1,200: at 1,200, 16 of 500 Phase 1 answers came back empty
    # because hidden reasoning used the whole cap (finish_reason="length").
    # A cap only costs money when it is reached; the median answer uses ~170.
    answer_max_tokens: int = 4000
    judge_max_tokens: int = 800
    # Reasoning effort for reasoning models ("minimal"/"low"/"medium"/"high").
    # Reasoning tokens are billed as output tokens, so "low" is the cost-effective
    # default for short, well-scoped prompts. Set to an empty string in `.env` if
    # the model is not a reasoning model. Both values are part of the cache key.
    # Answering and judging have separate settings so a phase can tune its answer
    # model without moving the judge.
    answer_reasoning_effort: str | None = "low"
    # DO NOT CHANGE for the rest of the series. Every phase is scored by the same
    # judge; changing this changes the verdicts (scores stop being comparable)
    # and invalidates every cached verdict (the whole evaluation is paid again).
    judge_reasoning_effort: str | None = "low"
    # Max tokens of retrieved context we will ever put in an answer prompt.
    # Beyond this, extra passages mostly add noise (and cost) for the generator.
    context_token_budget: int = 6000

    # -------------------------------------------------------------- retrieval
    # The leaderboard reports Recall@10, so every phase retrieves at most 10 docs
    # for the evaluator by default.
    recall_k: int = 10

    # ---------------------------------------------------------- curation
    noise_sources: list[str] = Field(
        default_factory=lambda: ["slack", "gmail", "confluence", "linear"]
    )
    noise_count: int = 4000
    seed: int = 42

    @cached_property
    def paths(self) -> Paths:
        """Filesystem locations (see `Paths`). Cached: computed once per process."""
        return Paths()


# A single shared instance. Import this rather than constructing `Settings()`
# yourself so that the `.env` file is parsed exactly once.
settings = Settings()

def setup_logging(level: int = 20) -> None:
    """Standard log format for every script; silences per-request HTTP chatter.

    httpx (used by the OpenAI and Qdrant clients) logs every request at INFO,
    which drowns out useful progress messages during a 40k-chunk ingest.
    """
    import logging

    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "urllib3", "typesafe_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# pydantic-settings reads `.env` into `settings` but does not export it to the
# process environment, while the OpenAI and TypeSafe SDKs look for their keys in
# `os.environ`. Export them once here (never overriding a real env var) so every
# SDK finds its key without each call site having to pass it explicitly.
for _var, _value in (("OPENAI_API_KEY", settings.openai_api_key), ("TYPESAFE_API_KEY", settings.typesafe_api_key)):
    if _value:
        os.environ.setdefault(_var, _value)
