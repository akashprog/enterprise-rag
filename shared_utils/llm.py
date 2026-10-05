"""Cost-aware wrappers around every paid model call in the series.

Every phase talks to paid APIs *only* through this module. That gives us, in
one place:

    1. Caching      - identical requests are answered from the SQLite cache
                      (shared_utils/cache.py) and cost $0.
    2. Usage & cost - every call's tokens are counted per model, cache hits are
                      counted separately, and a dollar figure is computed from
                      `settings.model_prices`. Each phase writes this report next
                      to its scores, so "better" always comes with "at what cost".
    3. Dry runs     - `dry_run=True` never calls the API; it records an
                      *upper-bound* token estimate instead so you can price a run
                      before paying for it.
    4. Budgets      - hard caps on output tokens (`max_tokens`) and helpers to
                      trim retrieved context to a token budget before it reaches
                      the prompt.
    5. Limits       - bounded concurrency (`run_limited`) plus the OpenAI
                      client's built-in exponential-backoff retries for 429/5xx.

Public API:
    usage                   - process-wide `UsageTracker` (tokens + cost report).
    count_tokens(text)      - tiktoken count, used for budgets and estimates.
    truncate_to_tokens()    - cut one text to N tokens.
    fit_to_budget()         - keep passages (in rank order) until a budget fills.
    CachedChat              - cached, tracked chat model (text or structured).
    CachedEmbeddings        - cached, batched, tracked embeddings; a drop-in
                              LangChain `Embeddings`, so vector stores can use it.
    run_limited()           - run many async calls with a concurrency cap.

The Jev (TypeSafe) wrapper is added here in Step 4, following the same
cache -> call -> track pattern.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence, TypeVar

import tiktoken
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ValidationError
from tqdm.asyncio import tqdm_asyncio

from shared_utils.cache import get_cache, make_key
from shared_utils.config import settings

# tiktoken downloads its tokenizer files on first use and, by default, keeps
# them in a temp dir the OS may wipe. Pinning the cache inside the repo's
# `.cache/` means one download ever, and offline dry runs afterwards.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(settings.paths.root / ".cache" / "tiktoken"))

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Usage & cost tracking
# =============================================================================
@dataclass
class ModelUsage:
    """Running totals for one model within one process.

    Attributes:
        calls:          Real API calls made (each one was paid for).
        cached_calls:   Requests answered from the cache (free).
        estimated_calls: Dry-run requests (nothing was sent).
        input_tokens / output_tokens: Tokens actually billed by real calls.
        estimated_input_tokens / estimated_output_tokens: Dry-run estimates.
            Output estimates assume the full `max_tokens` is used, so they are
            an upper bound -- real runs are usually cheaper.
    """

    calls: int = 0
    cached_calls: int = 0
    estimated_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0


class UsageTracker:
    """Thread-safe per-model token and cost accounting for one process run."""

    def __init__(self) -> None:
        self._by_model: dict[str, ModelUsage] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Clear all counters, e.g. between answering and judging so their costs are reported separately."""
        with self._lock:
            self._by_model.clear()

    def _get(self, model: str) -> ModelUsage:
        return self._by_model.setdefault(model, ModelUsage())

    def record_call(self, model: str, input_tokens: int, output_tokens: int, n_calls: int = 1) -> None:
        """Record `n_calls` real (billed) API requests and their total tokens."""
        with self._lock:
            u = self._get(model)
            u.calls += n_calls
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens

    def record_cache_hit(self, model: str, n: int = 1) -> None:
        """Record `n` requests served from the cache at zero cost."""
        with self._lock:
            self._get(model).cached_calls += n

    def record_estimate(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record a dry-run estimate (no API call was made)."""
        with self._lock:
            u = self._get(model)
            u.estimated_calls += 1
            u.estimated_input_tokens += input_tokens
            u.estimated_output_tokens += output_tokens

    @staticmethod
    def price(model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Dollar cost for a token count, or None if the model has no price set.

        Returning None (instead of guessing) keeps reports honest: a missing
        price shows up as "unknown" rather than a misleading $0.00.
        """
        p = settings.model_prices.get(model)
        if p is None and model.startswith("jev-"):
            # Jev bills one price across versions, so a pinned id like
            # `jev-1.13.0` may use the price listed under the `jev-latest` alias.
            p = settings.model_prices.get("jev-latest")
        if p is None:
            return None
        return (input_tokens * p.input + output_tokens * p.output) / 1_000_000

    def report(self) -> dict[str, Any]:
        """Snapshot of all usage with actual and estimated dollar costs.

        Returns:
            {"models": {model: {...counts, "cost_usd", "estimated_cost_usd"}},
             "total_cost_usd": float | None, "total_estimated_cost_usd": float | None}
            Totals are None when any model involved lacks a price.
        """
        with self._lock:
            snapshot = {m: ModelUsage(**vars(u)) for m, u in self._by_model.items()}

        models: dict[str, Any] = {}
        total: float | None = 0.0
        total_est: float | None = 0.0
        for model, u in snapshot.items():
            cost = self.price(model, u.input_tokens, u.output_tokens)
            est = self.price(model, u.estimated_input_tokens, u.estimated_output_tokens)
            models[model] = {**vars(u), "cost_usd": cost, "estimated_cost_usd": est}
            if u.calls:
                total = None if (total is None or cost is None) else total + cost
            if u.estimated_calls:
                total_est = None if (total_est is None or est is None) else total_est + est
        return {"models": models, "total_cost_usd": total, "total_estimated_cost_usd": total_est}

    def summary_lines(self) -> list[str]:
        """Human-readable report, one line per model plus totals."""

        def usd(x: float | None) -> str:
            return "unknown (set MODEL_PRICES)" if x is None else f"${x:,.4f}"

        rep = self.report()
        lines = []
        for model, u in rep["models"].items():
            line = (
                f"{model}: {u['calls']} paid calls ({u['input_tokens']:,} in / "
                f"{u['output_tokens']:,} out tok, {usd(u['cost_usd'])}), "
                f"{u['cached_calls']} cache hits"
            )
            if u["estimated_calls"]:
                line += (
                    f", DRY-RUN {u['estimated_calls']} calls ~{u['estimated_input_tokens']:,} in / "
                    f"<={u['estimated_output_tokens']:,} out tok, <= {usd(u['estimated_cost_usd'])}"
                )
            lines.append(line)
        lines.append(f"TOTAL actual cost: {usd(rep['total_cost_usd'])}")
        if any(u["estimated_calls"] for u in rep["models"].values()):
            lines.append(f"TOTAL dry-run upper bound: {usd(rep['total_estimated_cost_usd'])}")
        return lines

    def log_summary(self) -> None:
        """Log `summary_lines()` at INFO level."""
        for line in self.summary_lines():
            logger.info("[usage] %s", line)

    def write_json(self, path: Path) -> None:
        """Persist `report()` as JSON (e.g. results/p1_cost.json)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2))


# One tracker per process: every wrapper below reports into it.
usage = UsageTracker()


# =============================================================================
# Token counting & budgets
# =============================================================================
@lru_cache(maxsize=16)
def _encoding_for(model: str) -> tiktoken.Encoding:
    """Return the tokenizer for `model`, falling back to a modern default.

    tiktoken only knows model names it was released with. For newer models we
    fall back to `o200k_base` (the GPT-4o/GPT-5 family tokenizer); counts may be
    off by a few percent, which is fine for budgets and cost estimates.
    Cached so we build each tokenizer once per process.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str | None = None) -> int:
    """Count tokens in `text` for `model` (defaults to the answer model)."""
    return len(_encoding_for(model or settings.answer_model).encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int, model: str | None = None) -> str:
    """Return `text` cut to at most `max_tokens` tokens (unchanged if shorter)."""
    enc = _encoding_for(model or settings.answer_model)
    ids = enc.encode(text, disallowed_special=())
    return text if len(ids) <= max_tokens else enc.decode(ids[:max_tokens])


def fit_to_budget(
    passages: Sequence[str], budget: int | None = None, model: str | None = None
) -> list[str]:
    """Keep passages in rank order until `budget` tokens are used.

    The last passage that does not fully fit is truncated rather than dropped,
    so we use the budget completely. Everything after it is dropped: those are
    the lowest-ranked passages, i.e. the ones most likely to be noise.

    Args:
        passages: Retrieved texts, best first.
        budget:   Token budget (defaults to `settings.context_token_budget`).
        model:    Tokenizer to count with.

    Returns:
        The passages that fit (possibly with the last one truncated).
    """
    budget = settings.context_token_budget if budget is None else budget
    kept: list[str] = []
    used = 0
    for text in passages:
        n = count_tokens(text, model)
        if used + n <= budget:
            kept.append(text)
            used += n
            continue
        remaining = budget - used
        # Only keep a truncated tail if it is big enough to carry real content.
        if remaining > 50:
            kept.append(truncate_to_tokens(text, remaining, model))
        break
    return kept


# =============================================================================
# Chat models
# =============================================================================
# Step 2.0b raised the cap when reasoning consumed it and the answer came back
# empty (1,200 → 4,000). A parked answer was still empty at the 4,000 default,
# so the automatic retry is one step further, at 16,000.
ANSWER_EMPTY_RETRY_TOKENS = 16000
Message = tuple[str, str] | dict[str, str]


def _normalise_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    """Convert ("system", "...") tuples / role dicts into plain role dicts.

    A single canonical form matters for caching: the same conversation must
    always hash to the same key regardless of how the caller spelled it.
    "human" is mapped to "user" for the same reason.
    """
    out = []
    for m in messages:
        role, content = (m["role"], m["content"]) if isinstance(m, dict) else m
        out.append({"role": "user" if role == "human" else role, "content": content})
    return out


@dataclass
class ChatResult:
    """Outcome of one `CachedChat` request.

    Attributes:
        text:     The model's text reply ("" for structured calls and dry runs).
        parsed:   Structured output as a plain dict (structured calls only).
        input_tokens / output_tokens: Billed tokens (0 for cache hits).
        cached:   True if served from the cache (cost $0).
        dry_run:  True if no call was made (text/parsed are empty).
        finish_reason: Provider finish reason; "length" means `max_tokens` cut
            the reply short -- worth knowing because it silently hurts scores.
    """

    text: str = ""
    parsed: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    dry_run: bool = False
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CachedChat:
    """A chat model with caching, usage tracking, output caps and dry-run.

    Built on LangChain's `init_chat_model`, so any provider LangChain supports
    works by changing the model string (e.g. "anthropic:claude-...").

    Args:
        model:       Model name. OpenAI is assumed unless "provider:model" is given.
        max_tokens:  Hard cap on generated tokens (includes hidden reasoning
                     tokens on reasoning models, so do not set it too low).
        reasoning_effort: For reasoning models, lower effort = fewer (billed)
                     reasoning tokens. Defaults to `settings.answer_reasoning_effort`;
                     judges must pass `settings.judge_reasoning_effort`. An empty
                     string sends nothing (for non-reasoning models).
        dry_run:     If True, never call the API; record an estimate instead.

    Example:
        judge = CachedChat(settings.judge_model, max_tokens=400)
        res = await judge.ainvoke([("system", "..."), ("user", "...")],
                                  prompt_version="judge-v1", schema=Verdict)
        res.parsed["correct"]
    """

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int,
        reasoning_effort: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        # An empty string in .env means "not a reasoning model": send nothing.
        effort = reasoning_effort if reasoning_effort is not None else settings.answer_reasoning_effort
        self.reasoning_effort = effort or None
        self.dry_run = dry_run
        self._llm = None  # built lazily: dry runs and all-cache-hit runs need no API key

    def _client(self):
        """Create the underlying LangChain chat model on first real use."""
        if self._llm is None:
            from langchain.chat_models import init_chat_model

            kwargs: dict[str, Any] = {
                "max_tokens": self.max_tokens,
                # The OpenAI SDK retries 429/5xx/timeouts with exponential backoff.
                "max_retries": settings.max_retries,
            }
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            provider = None if ":" in self.model else "openai"
            self._llm = init_chat_model(self.model, model_provider=provider, **kwargs)
        return self._llm

    def _key(self, messages: list[dict[str, str]], prompt_version: str, schema: type[BaseModel] | None) -> str:
        # Everything that can change the output goes into the key. The schema's
        # JSON definition is included so editing a field invalidates old entries.
        schema_def = schema.model_json_schema() if schema else None
        return make_key(
            "chat", self.model, prompt_version, messages, schema_def, self.max_tokens, self.reasoning_effort
        )

    async def ainvoke(
        self,
        messages: Sequence[Message],
        *,
        prompt_version: str,
        schema: type[BaseModel] | None = None,
    ) -> ChatResult:
        """Send one request (or serve it from cache).

        Args:
            messages:       Conversation as ("role", "content") tuples or dicts.
            prompt_version: A label you bump whenever the prompt template changes,
                            so old cached replies are not reused for a new prompt.
            schema:         Optional Pydantic model for structured output. Using
                            structured output instead of "reply in JSON" prose is
                            both more reliable and cheaper (no retries on bad JSON).

        Returns:
            A `ChatResult`. Only real calls cost money; see `ChatResult.cached`.
        """
        msgs = _normalise_messages(messages)
        key = self._key(msgs, prompt_version, schema)
        cache = get_cache()

        hit = cache.get_json(key)
        if hit is not None:
            usage.record_cache_hit(self.model)
            cached = ChatResult(**hit, cached=True)
            return await self._retry_if_empty(messages, prompt_version, schema, cached)

        if self.dry_run:
            # Upper bound: assume the model uses its whole output allowance.
            est_in = sum(count_tokens(m["content"], self.model) for m in msgs)
            if schema is not None:
                est_in += count_tokens(json.dumps(schema.model_json_schema()), self.model)
            usage.record_estimate(self.model, est_in, self.max_tokens)
            return ChatResult(dry_run=True, input_tokens=est_in)

        llm = self._client()
        if schema is not None:
            # include_raw=True gives us the raw AIMessage too, which carries the
            # token usage we need for cost tracking.
            try:
                out = await llm.with_structured_output(schema, include_raw=True).ainvoke(msgs)
            except ValidationError as exc:
                # The OpenAI SDK parses structured replies itself and raises when
                # the JSON was cut off by `max_tokens`. Treat it as a failed parse
                # (not cached, caller falls back) instead of crashing the run.
                # Tokens of this call are billed but unknown here, so not recorded.
                logger.warning("%s structured reply unparseable (likely hit max_tokens=%d): %s",
                               self.model, self.max_tokens, str(exc).splitlines()[0])
                return ChatResult(finish_reason="length")
            raw, parsed_obj = out["raw"], out["parsed"]
            if out.get("parsing_error"):
                # First line only: the full error embeds the raw response, including
                # kilobytes of encrypted reasoning content.
                logger.warning("Structured output parse error (%s): %s", self.model,
                               str(out["parsing_error"]).splitlines()[0][:300])
            parsed = parsed_obj.model_dump() if parsed_obj is not None else None
            text = ""
        else:
            raw = await llm.ainvoke(msgs)
            parsed, text = None, raw.text if isinstance(raw.text, str) else raw.text()

        meta = getattr(raw, "usage_metadata", None) or {}
        in_tok, out_tok = int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
        finish = (getattr(raw, "response_metadata", None) or {}).get("finish_reason")
        usage.record_call(self.model, in_tok, out_tok)
        if finish == "length":
            logger.warning(
                "%s hit max_tokens=%d (reply truncated). Consider raising the cap.", self.model, self.max_tokens
            )

        result = ChatResult(text=text, parsed=parsed, input_tokens=in_tok, output_tokens=out_tok, finish_reason=finish)
        # Never cache a failed parse or an empty answer. An empty answer is the
        # reasoning-cap failure from Step 2.0b, and caching it would replay it.
        if (schema is not None and parsed is not None) or (schema is None and (result.text or "").strip()):
            cache.set_json(
                key,
                {
                    "text": result.text,
                    "parsed": result.parsed,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "finish_reason": finish,
                },
            )
        return await self._retry_if_empty(messages, prompt_version, schema, result)

    async def _retry_if_empty(self, messages, prompt_version: str, schema, result: ChatResult) -> ChatResult:
        """One higher-cap retry when an answer is empty. Structured calls are left alone."""
        if (
            schema is not None
            or self.dry_run
            or (result.text or "").strip()
            or self.max_tokens >= ANSWER_EMPTY_RETRY_TOKENS
        ):
            return result
        logger.warning(
            "Empty answer at max_tokens=%d; retrying at %d.",
            self.max_tokens, ANSWER_EMPTY_RETRY_TOKENS,
        )
        retry = CachedChat(
            self.model,
            max_tokens=ANSWER_EMPTY_RETRY_TOKENS,
            reasoning_effort=self.reasoning_effort,
            dry_run=False,
        )
        return await retry.ainvoke(messages, prompt_version=prompt_version)


# =============================================================================
# Embeddings
# =============================================================================
class CachedEmbeddings(Embeddings):
    """Batched, cached, usage-tracked embeddings (a LangChain `Embeddings`).

    Because it subclasses LangChain's `Embeddings`, it plugs straight into
    `QdrantVectorStore` and friends, and every vector they request goes through
    the cache -- re-ingesting a collection costs $0 for texts already embedded.

    Cost behaviour:
        * Cache lookup is one bulk SQLite query per call.
        * Only *unique, uncached* texts are sent (duplicates within a call, e.g.
          boilerplate email footers, are embedded once).
        * Misses are sent in batches of `settings.embedding_batch_size`.

    Args:
        model: OpenAI embedding model name.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.embedding_model
        self._client = None

    def _openai(self):
        """Create the LangChain OpenAI embeddings client on first real use."""
        if self._client is None:
            from langchain_openai import OpenAIEmbeddings

            self._client = OpenAIEmbeddings(
                model=self.model,
                chunk_size=settings.embedding_batch_size,  # texts per API request
                max_retries=settings.max_retries,
            )
        return self._client

    def _key(self, text: str) -> str:
        return make_key("embed", self.model, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, paying only for ones never embedded before."""
        cache = get_cache()
        keys = [self._key(t) for t in texts]
        found = cache.get_vectors(keys)

        # Unique misses only, preserving first-seen order.
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in found and k not in missing:
                missing[k] = t
        if len(found):
            usage.record_cache_hit(self.model, sum(1 for k in keys if k in found))

        if missing:
            miss_keys, miss_texts = list(missing), list(missing.values())
            vectors = self._openai().embed_documents(miss_texts)
            # The embeddings endpoint's token usage is not surfaced by LangChain,
            # so we count locally with the model's tokenizer (exact for OpenAI).
            n_tok = sum(count_tokens(t, self.model) for t in miss_texts)
            n_requests = -(-len(miss_texts) // settings.embedding_batch_size)  # ceil division
            usage.record_call(self.model, n_tok, 0, n_calls=n_requests)
            new = dict(zip(miss_keys, vectors))
            cache.set_vectors(new.items())
            found.update(new)

        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query (cached, so repeated evaluation queries are free)."""
        return self.embed_documents([text])[0]

    def estimate_cost(self, texts: Iterable[str]) -> dict[str, Any]:
        """Price embedding `texts` without calling the API (for --dry-run).

        Already-cached texts are excluded, so the estimate is what you would
        actually pay right now.
        """
        texts = list(texts)
        keys = [self._key(t) for t in texts]
        unique = dict(zip(keys, texts))
        cached = get_cache().get_vectors(list(unique))
        todo = {k: t for k, t in unique.items() if k not in cached}
        n_tok = sum(count_tokens(t, self.model) for t in todo.values())
        return {
            "texts": len(texts),
            # Identical texts (e.g. repeated email signatures) are embedded once.
            "duplicates": len(texts) - len(unique),
            "already_cached": len(cached),
            "to_embed": len(todo),
            "tokens": n_tok,
            "cost_usd": UsageTracker.price(self.model, n_tok, 0),
        }


# =============================================================================
# Concurrency
# =============================================================================
async def run_limited(
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    limit: int | None = None,
    desc: str = "",
) -> list[R]:
    """Apply async `fn` to every item with at most `limit` calls in flight.

    Unbounded `asyncio.gather` over 500 questions would fire 500 requests at
    once and trigger rate-limit errors (each retry = wasted time). A semaphore
    keeps us at a steady, provider-friendly rate. Results keep input order.

    Args:
        items: Inputs to process.
        fn:    Async function applied to each item.
        limit: Max concurrency (defaults to `settings.max_concurrency`).
        desc:  Progress-bar label.
    """
    sem = asyncio.Semaphore(limit or settings.max_concurrency)

    async def guarded(item: T) -> R:
        async with sem:
            return await fn(item)

    return await tqdm_asyncio.gather(*(guarded(i) for i in items), desc=desc, disable=not desc)
