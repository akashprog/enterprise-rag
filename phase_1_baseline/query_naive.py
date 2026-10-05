"""Phase 1 query: top-10 cosine similarity + one LLM call. The baseline to beat.

Pipeline per question (what nearly every RAG tutorial does):
    1. Embed the question (text-embedding-3-small, cached).
    2. Cosine-similarity search over `p1_naive`, top k=10 chunks.
    3. Paste those chunks into a standard "answer from the context" prompt and
       ask the answer model once.
    4. Report the documents behind the 10 chunks as `doc_ids` (de-duplicated,
       best rank first). Nothing is filtered, so every loosely similar chunk
       counts as a retrieved document -- expect many Invalid Extra Docs.

Why it is expected to struggle on this benchmark:
    * Conflicting Info: two near-duplicate docs with different facts both rank
      high and the prompt has no way to tell which is current.
    * Completeness / Project Related: answers spread over 4-10 docs cannot fit
      in 10 chunks when several chunks come from the same doc.
    * Codenames / ticket IDs: dense embeddings blur exact identifiers.

Cost per question: one cached query embedding (~$0.0000003) plus one answer
call (~2.5k input tokens of context, output capped at `answer_max_tokens`).
Use --dry-run to price a run first; repeated runs are served from the cache.

Usage:
    python -m phase_1_baseline.query_naive --dry-run
    python -m phase_1_baseline.query_naive --limit 20 --evaluate
    python -m phase_1_baseline.query_naive --evaluate            # all 500 questions + scoring

Reads:  Qdrant collection `p1_naive`, data/raw_onyx_subset/mini_redwood_qa.jsonl
Writes: results/phase_1/fixed/  (answers.jsonl, answer_cost.json, <judge>/ with --evaluate)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_qdrant import QdrantVectorStore  # noqa: E402

from phase_1_baseline.ingest_naive import COLLECTION  # noqa: E402
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.llm import CachedChat, CachedEmbeddings, count_tokens, fit_to_budget, usage  # noqa: E402
from shared_utils.runner import add_common_args, run_phase  # noqa: E402
from shared_utils.vectorstore import dense_store, doc_ids_from, get_client  # noqa: E402

logger = logging.getLogger("phase1.query")

TOP_K = 10
PROMPT_VERSION = "p1-naive-v1"

# A standard tutorial-style RAG prompt. Deliberately plain: no source
# attribution, no guidance on conflicts or dates. Those come in later phases.
SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved "
    "context to answer the question. If you don't know the answer, say that you don't know."
)
USER_PROMPT = "Question: {question}\n\nContext:\n{context}\n\nAnswer:"

# Rough size of one 1,000-character chunk in tokens; only used for dry-run estimates.
_EST_TOKENS_PER_CHUNK = 250


class NaivePipeline:
    """Holds the (lazily created) vector store and answer model for Phase 1.

    Args:
        dry_run: If True, no paid API is called; costs are estimated instead.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.embeddings = CachedEmbeddings()
        self.llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens, dry_run=dry_run)
        self._store: QdrantVectorStore | None = None

    @property
    def store(self) -> QdrantVectorStore:
        if self._store is None:
            client = get_client()
            if not client.collection_exists(COLLECTION):
                raise RuntimeError(f"Collection {COLLECTION} missing. Run phase_1_baseline.ingest_naive first.")
            self._store = dense_store(client, COLLECTION, self.embeddings)
        return self._store

    def _estimate(self, question: str) -> dict[str, Any]:
        """Dry run: record upper-bound costs without embedding or retrieving."""
        usage.record_estimate(self.embeddings.model, count_tokens(question, self.embeddings.model), 0)
        prompt_tokens = count_tokens(SYSTEM_PROMPT + USER_PROMPT + question) + TOP_K * _EST_TOKENS_PER_CHUNK
        usage.record_estimate(self.llm.model, min(prompt_tokens, settings.context_token_budget + 200),
                              self.llm.max_tokens)
        return {"answer": "", "doc_ids": []}

    async def answer(self, question: str) -> dict[str, Any]:
        """Retrieve top-k chunks by cosine similarity and answer from them."""
        if self.dry_run:
            return self._estimate(question)

        # The Qdrant client and our embedding cache are synchronous; running the
        # search in a worker thread keeps other questions' LLM calls flowing.
        hits = await asyncio.to_thread(self.store.similarity_search, question, TOP_K)
        context = "\n\n".join(fit_to_budget([h.page_content for h in hits]))

        res = await self.llm.ainvoke(
            [("system", SYSTEM_PROMPT), ("user", USER_PROMPT.format(question=question, context=context))],
            prompt_version=PROMPT_VERSION,
        )
        return {"answer": res.text, "doc_ids": doc_ids_from(hits)}


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 1: naive top-k cosine RAG over p1_naive.")
    add_common_args(p)
    args = p.parse_args()
    setup_logging()
    try:
        pipeline = NaivePipeline(dry_run=args.dry_run)
        # Official Phase 1 baseline. A different corpus or config belongs in its
        # own directory, not here, so this one stays comparable.
        return run_phase("p1", pipeline.answer, args,
                         out_dir=settings.paths.results_dir / "phase_1" / "fixed")
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
