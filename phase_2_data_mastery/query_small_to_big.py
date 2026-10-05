"""Phase 2, Step 2.1: small-to-big context. Match on small chunks, answer from bigger context.

The one change against Phase 1
    Matching is exactly Phase 1's: the same `p1_naive` collection, the same
    embeddings, the same cosine top-k chunk search (just k=20 instead of 10).
    The prompt is exactly Phase 1's too. Only the CONTEXT sent to the model changes.

Why
    On Phase 1's wrong answers whose gold documents were ALL retrieved, the prompt
    held only 34% of the gold facts on average (61% for correct answers): the
    right document was found, but the 1,000-character chunk that matched the
    question was often not the chunk holding the answer. Small chunks are good
    for matching (one topic per vector); answers need the surrounding text.

Context rule, per question
    1. Search the top `--top-chunks` (20) chunks.
    2. Group them by document, documents ordered by their best chunk.
    3. For each of the top `--docs` (N=5) documents:
         * "whole"  if the document has <= `--whole-doc-max` tokens, send all of it;
         * "window" otherwise: the matched chunks plus `--window` (W) neighbouring
           chunks on each side, merged into contiguous spans so no text repeats.
           Gaps between spans are marked with "[...]".
    4. Add documents in that order until `--budget` tokens of context are used.
       A document that does not fit is shrunk rather than skipped: a whole
       document falls back to its window ("whole->window"), and a window that
       still does not fit is cut to the remaining budget ("...+cut"), after
       which no more documents are added. A cut smaller than MIN_PARTIAL_TOKENS
       is not worth sending, so then the document is left out.
    The documents reported as retrieved (`doc_ids`) are the documents actually
    sent, in that order, so retrieval metrics describe what the model saw.

Text fidelity
    Documents are re-split locally with Phase 1's splitter, keeping each chunk's
    character offset, so a window is an exact slice of the original document and
    overlapping chunks (200 characters) are never pasted twice. Every Qdrant hit
    is checked against the local split (`chunk_mismatches` in the summary).

Cost
    Retrieval is free: question embeddings are cached from Phase 1 and Qdrant is
    local. Each question is one answer call, like Phase 1, but with a bigger
    context (more input tokens; input is the cheap side of the price).
    --contexts-only builds every context and prices it with no paid call at all.
    --dry-run prices exactly (real contexts), counting cached answers as free.

Usage:
    python -m phase_2_data_mastery.query_small_to_big --contexts-only                      # free
    python -m phase_2_data_mastery.query_small_to_big --contexts-only --window 2 --budget 10000
    python -m phase_2_data_mastery.query_small_to_big --dry-run
    python -m phase_2_data_mastery.query_small_to_big --evaluate                           # paid
    python -m phase_2_data_mastery.query_small_to_big --plain-chunks --top-chunks 40 --budget 5400  # A/B control

Reads:  Qdrant `p1_naive`, data/raw_onyx_subset/mini_redwood_{docs,qa}.jsonl
Writes: --contexts-only: results/phase_2/contexts/<setting>/
        otherwise:       results/phase_2/<setting>/ (answers, cost, <judge>/ with --evaluate)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from phase_1_baseline.ingest_naive import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, load_docs  # noqa: E402
from phase_1_baseline.query_naive import SYSTEM_PROMPT, USER_PROMPT  # noqa: E402
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils import evaluation  # noqa: E402
from shared_utils.evaluation import load_qa  # noqa: E402
from shared_utils.llm import CachedChat, CachedEmbeddings, count_tokens, fit_to_budget, truncate_to_tokens  # noqa: E402
from shared_utils.runner import add_common_args, run_phase  # noqa: E402
from shared_utils.vectorstore import dense_store, get_client  # noqa: E402

logger = logging.getLogger("phase2.small_to_big")

# The prompt text is Phase 1's (imported above); the version label is separate
# only so Phase 2 answers are easy to tell apart in the cache.
PROMPT_VERSION = "p2-small-to-big-v1"
PHASE = "p2"

DEFAULT_TOP_CHUNKS = 20
DEFAULT_DOCS = 5
DEFAULT_WINDOW = 1
# From measure_doc_lengths.py: 77% of all documents (75% of gold documents) are
# <= 2,000 tokens, and 5 such documents fit a ~10k budget. Longer documents
# (most Confluence pages, long Drive docs, meeting transcripts) get a window.
DEFAULT_WHOLE_DOC_MAX = 2000
DEFAULT_BUDGET = 8000
# A document cut to fewer tokens than this is mostly noise; leave it out instead.
MIN_PARTIAL_TOKENS = 200

DOC_SEPARATOR = "\n\n"  # between documents: the same join Phase 1 uses between chunks
GAP_MARKER = "\n\n[...]\n\n"  # between non-adjacent spans of one document


# =============================================================================
# Documents, re-split exactly as Phase 1 ingested them
# =============================================================================
@dataclass
class SplitDoc:
    """One document as Phase 1 chunked it, with each chunk's character span.

    Attributes:
        text:   The exact text Phase 1 split ("title\\n\\nbody").
        spans:  (start, end) character offsets of chunk i in `text`.
        tokens: Token count of the whole `text`.
    """

    doc_id: str
    text: str
    spans: list[tuple[int, int]]
    tokens: int

    def chunk(self, i: int) -> str:
        start, end = self.spans[i]
        return self.text[start:end]

    def window(self, matched: list[int], width: int) -> str:
        """Matched chunks plus `width` neighbours each side, merged, in document order.

        Neighbouring chunks overlap by up to 200 characters, so spans are merged
        by character offset: each piece of text appears once.
        """
        last = len(self.spans) - 1
        ranges = sorted((max(0, i - width), min(last, i + width)) for i in set(matched))
        merged: list[list[int]] = []
        for lo, hi in ranges:
            if merged and lo <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return GAP_MARKER.join(self.text[self.spans[lo][0]:self.spans[hi][1]] for lo, hi in merged)


class DocIndex:
    """Lazily re-splits mini-redwood documents with Phase 1's splitter, keyed by path.

    Path, not doc_id, identifies a document: 4 doc_ids in the corpus are shared
    by two different files (see shared_utils/vectorstore.py).
    """

    def __init__(self) -> None:
        docs = load_docs(None)
        self._rows = {r.path: r for r in docs.itertuples(index=False)}
        self._split: dict[str, SplitDoc] = {}
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)

    def source_of_path(self, path: str) -> str:
        """Source type (slack, gmail, ...) of the file at `path`."""
        return self._rows[path].source_type

    def get(self, path: str) -> SplitDoc:
        if path not in self._split:
            row = self._rows[path]
            text = f"{row.title}\n\n{row.content}"
            pieces = self._splitter.create_documents([text])
            spans = [(p.metadata["start_index"], p.metadata["start_index"] + len(p.page_content)) for p in pieces]
            self._split[path] = SplitDoc(row.doc_id, text, spans, count_tokens(text))
        return self._split[path]


# =============================================================================
# Context building
# =============================================================================
@dataclass
class ContextSettings:
    """The four knobs of the expansion rule (plus the search depth).

    `plain=True` is the A/B control arm: no expansion at all. It is Phase 1's
    exact context (top chunks in rank order, `fit_to_budget`, joined by a blank
    line) with only the chunk count and budget raised, so an A/B at equal prompt
    size isolates the small-to-big method from "more tokens". `docs`, `window`
    and `whole_doc_max` are ignored in that mode.
    """

    top_chunks: int = DEFAULT_TOP_CHUNKS
    docs: int = DEFAULT_DOCS
    window: int = DEFAULT_WINDOW
    whole_doc_max: int = DEFAULT_WHOLE_DOC_MAX
    budget: int = DEFAULT_BUDGET
    plain: bool = False

    @property
    def tag(self) -> str:
        """Short, file-name-safe label, e.g. "n5_w1_t2000_b8000" or "plain_k40_b5400"."""
        if self.plain:
            return f"plain_k{self.top_chunks}_b{self.budget}"
        return f"n{self.docs}_w{self.window}_t{self.whole_doc_max}_b{self.budget}"


@dataclass
class Passage:
    """What one document (or, in plain mode, one chunk) contributed to the prompt."""

    doc_id: str
    mode: str  # "whole", "window", "whole->window", "chunk", optionally + "+cut"
    matched_chunks: list[int]
    doc_tokens: int
    tokens: int
    text: str


@dataclass
class Context:
    """The full context for one question."""

    passages: list[Passage] = field(default_factory=list)
    candidates: int = 0  # documents the rule wanted to send (top N), before the budget
    chunk_mismatches: int = 0  # Qdrant hits whose text differs from the local re-split

    @property
    def text(self) -> str:
        return DOC_SEPARATOR.join(p.text for p in self.passages)

    @property
    def doc_ids(self) -> list[str]:
        """Documents sent, de-duplicated, in prompt order (plain mode has several chunks per doc)."""
        return list(dict.fromkeys(p.doc_id for p in self.passages))


def passage_for(doc: SplitDoc, matched: list[int], *, window: int, whole_doc_max: int) -> tuple[str, str, int]:
    """The passage arm A sends for one document, before the token budget is applied.

    A document of at most `whole_doc_max` tokens is sent whole. A longer one is
    sent as the matched chunks plus `window` neighbours on each side, merged so
    overlapping chunks are not repeated. Returns (mode, text, tokens) with mode
    "whole" or "window". The budget logic in `build_context` may still shrink it.
    """
    if doc.tokens <= whole_doc_max:
        return "whole", doc.text, doc.tokens
    text = doc.window(matched, window)
    return "window", text, count_tokens(text)


def build_plain_context(hits: list, index: DocIndex, cfg: ContextSettings) -> Context:
    """Control arm: Phase 1's context recipe, just with more chunks and a bigger budget."""
    kept = fit_to_budget([h.page_content for h in hits], cfg.budget)
    ctx = Context(candidates=len(hits))
    for h, text in zip(hits, kept):
        doc = index.get(h.metadata["path"])
        mode = "chunk" if text == h.page_content else "chunk+cut"
        ctx.passages.append(Passage(doc.doc_id, mode, [int(h.metadata["chunk_index"])], doc.tokens,
                                    count_tokens(text), text))
    return ctx


def build_context(hits: list, index: DocIndex, cfg: ContextSettings) -> Context:
    """Apply the small-to-big rule (module docstring) to ranked chunk hits."""
    if cfg.plain:
        return build_plain_context(hits, index, cfg)
    # Group chunk hits by document, in order of each document's best chunk.
    matched: dict[str, list[int]] = {}
    mismatches = 0
    for h in hits:
        path, i = h.metadata["path"], int(h.metadata["chunk_index"])
        matched.setdefault(path, []).append(i)
        doc = index.get(path)
        mismatches += i >= len(doc.spans) or doc.chunk(i) != h.page_content

    ctx = Context(candidates=min(cfg.docs, len(matched)), chunk_mismatches=mismatches)
    used = 0
    for path in list(matched)[: cfg.docs]:
        doc, chunks = index.get(path), matched[path]
        mode, text, n = passage_for(doc, chunks, window=cfg.window, whole_doc_max=cfg.whole_doc_max)

        remaining = cfg.budget - used
        if n > remaining and mode == "whole":
            text = doc.window(chunks, cfg.window)
            mode, n = "whole->window", count_tokens(text)
        if n > remaining:
            if remaining < MIN_PARTIAL_TOKENS:
                break
            text = truncate_to_tokens(text, remaining)
            mode, n = mode + "+cut", count_tokens(text)
        ctx.passages.append(Passage(doc.doc_id, mode, sorted(set(chunks)), doc.tokens, n, text))
        used += n
        if used >= cfg.budget or mode.endswith("+cut"):
            break
    return ctx


def messages(question: str, context: str) -> list[tuple[str, str]]:
    """Phase 1's prompt, unchanged, around the new context."""
    return [("system", SYSTEM_PROMPT), ("user", USER_PROMPT.format(question=question, context=context))]


# =============================================================================
# Pipeline
# =============================================================================
class SmallToBigPipeline:
    """Phase 1 retrieval + small-to-big context + Phase 1 prompt.

    Args:
        cfg:     Expansion rule settings.
        dry_run: Price answer calls instead of making them (contexts are real).
    """

    def __init__(self, cfg: ContextSettings, *, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.index = DocIndex()
        self.llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens, dry_run=dry_run)
        self._store = None

    @property
    def store(self):
        if self._store is None:
            client = get_client()
            if not client.collection_exists(COLLECTION):
                raise RuntimeError(f"Collection {COLLECTION} missing. Run phase_1_baseline.ingest_naive first.")
            self._store = dense_store(client, COLLECTION, CachedEmbeddings())
        return self._store

    def context(self, question: str) -> Context:
        """Retrieve (free: cached embedding, local Qdrant) and build the context."""
        return build_context(self.store.similarity_search(question, self.cfg.top_chunks), self.index, self.cfg)

    async def answer(self, question: str) -> dict[str, Any]:
        ctx = await asyncio.to_thread(self.context, question)
        res = await self.llm.ainvoke(messages(question, ctx.text), prompt_version=PROMPT_VERSION)
        return {"answer": res.text, "doc_ids": ctx.doc_ids}


# =============================================================================
# --contexts-only: the free check
# =============================================================================
def phase1_output_tokens_per_answer() -> float | None:
    """Average billed output tokens per answer in Phase 1 (fixed), for cost estimates.

    Prompt changes move input tokens; output length is assumed to stay as in
    Phase 1. It is an estimate, which is why the summary also gives an upper bound.
    All billed output tokens (the main run plus the --only-empty re-run) are
    divided by the number of questions, so the 16 wasted empty attempts make it
    slightly conservative.
    """
    fixed = settings.paths.results_dir / "phase_1" / "fixed"
    main_run = fixed / "answer_cost.json"
    if not main_run.is_file():
        return None
    questions = json.loads(main_run.read_text())["questions"]
    total_out = 0
    for path in (main_run, fixed / "answer_cost_only_empty.json"):
        if path.is_file():
            total_out += json.loads(path.read_text())["models"].get(settings.answer_model, {}).get("output_tokens", 0)
    return total_out / questions


def write_contexts(pipeline: SmallToBigPipeline, qa, out_dir: Path) -> dict[str, Any]:
    """Build every question's context, write it, and return the summary (no paid calls)."""
    cfg = pipeline.cfg
    dest = out_dir / cfg.tag
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "contexts.jsonl"
    prompt_tokens: list[int] = []
    modes: dict[str, int] = {}
    mismatches = docs_sent = 0
    with path.open("w", encoding="utf-8") as f:
        for q in qa.itertuples():
            ctx = pipeline.context(q.question)
            n_prompt = sum(count_tokens(text) for _, text in messages(q.question, ctx.text))
            prompt_tokens.append(n_prompt)
            mismatches += ctx.chunk_mismatches
            docs_sent += len(ctx.doc_ids)
            for p in ctx.passages:
                modes[p.mode] = modes.get(p.mode, 0) + 1
            # `answer` is empty so this file doubles as an answers file for
            # `shared_utils.evaluation --no-llm` (retrieval metrics on the docs sent).
            f.write(json.dumps({
                "question_id": q.question_id, "answer": "", "doc_ids": ctx.doc_ids,
                "prompt_tokens": n_prompt, "candidates": ctx.candidates,
                "passages": [asdict(p) for p in ctx.passages],
            }, ensure_ascii=False) + "\n")

    price = settings.model_prices.get(settings.answer_model)
    out_per_answer = phase1_output_tokens_per_answer()
    n = len(prompt_tokens)
    summary: dict[str, Any] = {
        "setting": asdict(cfg), "tag": cfg.tag, "questions": n,
        "avg_prompt_tokens": round(statistics.mean(prompt_tokens)),
        "p90_prompt_tokens": sorted(prompt_tokens)[int(0.9 * (n - 1))],
        "avg_docs_sent": round(docs_sent / n, 2),
        "avg_passages_sent": round(sum(modes.values()) / n, 2),
        "passages_by_mode": dict(sorted(modes.items(), key=lambda kv: -kv[1])),
        "chunk_mismatches": mismatches,
        "est_output_tokens_per_answer": None if out_per_answer is None else round(out_per_answer),
    }
    if price is not None:
        input_usd = sum(prompt_tokens) * price.input / 1e6
        summary["est_answer_cost_usd"] = (None if out_per_answer is None
                                          else round(input_usd + n * out_per_answer * price.output / 1e6, 4))
        summary["max_answer_cost_usd"] = round(input_usd + n * settings.answer_max_tokens * price.output / 1e6, 4)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", path)
    return summary


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2.1: Phase 1 retrieval, small-to-big context.")
    add_common_args(p)
    p.add_argument("--top-chunks", type=int, default=DEFAULT_TOP_CHUNKS, help="Chunks to search (Phase 1: 10).")
    p.add_argument("--docs", type=int, default=DEFAULT_DOCS, help="Max documents in the context (N).")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help="Neighbouring chunks added on each side of a matched chunk (W).")
    p.add_argument("--whole-doc-max", type=int, default=DEFAULT_WHOLE_DOC_MAX,
                   help="Documents with at most this many tokens are sent whole.")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Max context tokens.")
    p.add_argument("--plain-chunks", action="store_true",
                   help="A/B control: Phase 1's context (top chunks in rank order) with --top-chunks and "
                        "--budget raised; no expansion.")
    p.add_argument("--contexts-only", action="store_true",
                   help="Build and save every context and its estimated cost; no paid calls.")
    p.add_argument("--contexts-dir", type=Path, default=settings.paths.results_dir / "phase_2" / "contexts")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    cfg = ContextSettings(args.top_chunks, args.docs, args.window, args.whole_doc_max, args.budget,
                          plain=args.plain_chunks)
    try:
        pipeline = SmallToBigPipeline(cfg, dry_run=args.dry_run)
        if args.contexts_only:
            qa = load_qa()
            types = evaluation.selected_question_types(args)
            if types:
                qa = qa[qa["question_type"].isin(types)]
            if args.limit:
                qa = qa.head(args.limit)
            print(json.dumps(write_contexts(pipeline, qa, args.contexts_dir), indent=2))
            return 0
        logger.info("Context setting: %s", cfg.tag)
        # One directory per setting, so A/B arms never overwrite each other.
        return run_phase(cfg.tag, pipeline.answer, args,
                         out_dir=settings.paths.results_dir / "phase_2" / cfg.tag)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
