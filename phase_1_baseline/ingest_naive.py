"""Phase 1 ingest: the textbook LangChain recipe, applied to messy enterprise data.

This is deliberately the approach most tutorials teach, with no enterprise
awareness at all:

    1. Treat every document as one flat string (title line + body, exactly as
       the exported .txt file reads -- what LangChain's TextLoader would give).
    2. Split it with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
    3. Embed every chunk with `text-embedding-3-small` and store it in Qdrant
       (collection `p1_naive`, cosine distance).

What this ignores (and later phases fix):
    * Slack threads and Gmail reply chains are cut mid-conversation, so a chunk
      often holds an answer without the question it answers (Phase 2 loaders).
    * A 1,000-character chunk loses the document's title, channel and project
      context, so near-identical chunks from different projects look the same
      to the embedder (Phase 2 parent-child chunks).
    * Pure dense similarity misses exact codenames, ticket IDs and acronyms
      (Phase 3 BM25 hybrid search).

Cost: one-off embedding of ~42k chunks (~7.6M tokens including the 20% overlap),
about $0.15 with text-embedding-3-small (check with --dry-run). Re-runs are free: chunk IDs are deterministic,
existing points are skipped, and embeddings are cached.

Usage:
    python -m phase_1_baseline.ingest_naive --dry-run     # chunk count + cost estimate, no API calls
    python -m phase_1_baseline.ingest_naive --limit 50    # small trial
    python -m phase_1_baseline.ingest_naive               # full mini-redwood ingest
    python -m phase_1_baseline.ingest_naive --recreate    # drop and rebuild the collection

Reads:  data/raw_onyx_subset/mini_redwood_docs.jsonl
Writes: Qdrant collection `p1_naive`
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.llm import CachedEmbeddings, usage  # noqa: E402
from shared_utils.vectorstore import (  # noqa: E402
    chunk_id,
    dense_store,
    ensure_dense_collection,
    get_client,
    upsert_new_chunks,
)

logger = logging.getLogger("phase1.ingest")

COLLECTION = "p1_naive"
# The two most common values in LangChain tutorials. Characters, not tokens:
# 1,000 chars is roughly 250 tokens of English text.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Part of every chunk ID: change it whenever the chunking above changes, then
# run with --recreate so stale chunks are dropped.
CHUNKER_VERSION = f"p1-recursive-{CHUNK_SIZE}-{CHUNK_OVERLAP}"


def load_docs(limit: int | None) -> pd.DataFrame:
    """Load the curated mini-redwood documents (optionally only the first `limit`)."""
    path = settings.paths.mini_docs
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run scripts/curate_mini_redwood.py first.")
    docs = pd.read_json(path, lines=True)
    return docs.head(limit) if limit else docs


def chunk_documents(docs: pd.DataFrame) -> tuple[list[Document], list[str]]:
    """Split every document into naive fixed-size chunks.

    Returns:
        (chunks, ids): LangChain Documents with metadata, and their deterministic
        Qdrant point IDs (aligned by position).

    Metadata is carried only so we can map chunks back to documents for
    scoring. The naive pipeline does not use it for retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[Document] = []
    ids: list[str] = []
    for row in docs.itertuples(index=False):
        # The raw file as a tutorial loader would read it: title line, blank line, body.
        text = f"{row.title}\n\n{row.content}"
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append(Document(
                page_content=piece,
                metadata={"doc_id": row.doc_id, "source_type": row.source_type, "title": row.title,
                          "path": row.path, "chunk_index": i},
            ))
            ids.append(chunk_id(CHUNKER_VERSION, row.path, i))
    return chunks, ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: naive chunk + embed ingest into Qdrant.")
    p.add_argument("--limit", type=int, default=None, help="Ingest only the first N documents.")
    p.add_argument("--dry-run", action="store_true", help="Count chunks and estimate cost; no API calls.")
    p.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection first.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        docs = load_docs(args.limit)
        chunks, ids = chunk_documents(docs)
        logger.info("%d documents -> %d chunks (avg %.1f per doc).", len(docs), len(chunks), len(chunks) / len(docs))

        embeddings = CachedEmbeddings()
        if args.dry_run:
            est = embeddings.estimate_cost(c.page_content for c in chunks)
            logger.info("Dry run: %s", est)
            return 0

        client = get_client()
        ensure_dense_collection(client, COLLECTION, embeddings, recreate=args.recreate)
        added = upsert_new_chunks(dense_store(client, COLLECTION, embeddings), client, COLLECTION, chunks, ids)
        total = client.count(COLLECTION, exact=True).count
        logger.info("Added %d chunks; collection %s now holds %d points.", added, COLLECTION, total)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    finally:
        usage.log_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
