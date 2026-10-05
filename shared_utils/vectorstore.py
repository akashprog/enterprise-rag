"""Qdrant helpers shared by every phase: connection, collections, idempotent ingest.

Why a shared module?
    Every phase ingests chunks into its own Qdrant collection (`p1_naive`,
    `p2_parent_child`, `p3_hybrid`). The mechanics are identical -- connect,
    create the collection if needed, give each chunk a stable ID, skip chunks
    already stored, upsert the rest -- so they live here once.

Cost-relevant design:
    * Deterministic point IDs. Each chunk's ID is a UUIDv5 of
      (chunker version, document path, chunk index). Re-running an ingest
      produces the same IDs, so `upsert_new_chunks` can ask Qdrant which IDs
      already exist and embed *only* the missing ones. An interrupted ingest
      resumes where it stopped instead of starting over.
    * `path`, not `doc_id`, identifies a document: 4 doc_ids in the corpus are
      shared by two different files (see scripts/curate_mini_redwood.py).
    * Embeddings go through `CachedEmbeddings`, so even a `--recreate` re-ingest
      pays $0 for texts embedded before.

Payload layout (LangChain's QdrantVectorStore convention):
    {"page_content": <chunk text>, "metadata": {doc_id, source_type, title, path, chunk_index, ...}}

Reads/writes: the Qdrant server at `settings.qdrant_url` (see docker-compose.yml).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from shared_utils.config import settings
from shared_utils.llm import CachedEmbeddings

logger = logging.getLogger(__name__)

# Fixed namespace so chunk IDs are identical across machines and runs.
_ID_NAMESPACE = uuid.UUID("6f1c2a52-6c1e-4f3a-9a57-2f4a3f0e9b10")


def get_client() -> QdrantClient:
    """Connect to Qdrant, failing with a helpful message if it is not running."""
    client = QdrantClient(url=settings.qdrant_url, timeout=60)
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 - connection errors vary by transport
        raise RuntimeError(
            f"Cannot reach Qdrant at {settings.qdrant_url}. Start it with `docker compose up -d`."
        ) from exc
    return client


def chunk_id(chunker_version: str, path: str, chunk_index: int) -> str:
    """Stable UUID for one chunk.

    The chunker version is part of the ID so that changing chunking parameters
    yields new IDs (old chunks are never mistaken for new ones). Use
    `--recreate` after such a change to drop the stale points.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, f"{chunker_version}|{path}|{chunk_index}"))


def ensure_dense_collection(
    client: QdrantClient, name: str, embeddings: CachedEmbeddings, *, recreate: bool = False
) -> None:
    """Create a single-dense-vector cosine collection if it does not exist.

    The vector size is discovered by embedding one short probe string (cached
    after the first run, so free) rather than hard-coded, which keeps the code
    correct if you switch embedding models in `.env`.
    """
    if recreate and client.collection_exists(name):
        logger.info("Dropping collection %s (--recreate).", name)
        client.delete_collection(name)
    if client.collection_exists(name):
        return
    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    logger.info("Created collection %s (dim=%d, cosine).", name, dim)


def existing_ids(client: QdrantClient, collection: str, ids: Sequence[str], batch: int = 1000) -> set[str]:
    """Return the subset of `ids` already stored in `collection` (no vectors fetched)."""
    found: set[str] = set()
    for start in range(0, len(ids), batch):
        points = client.retrieve(
            collection, ids=list(ids[start : start + batch]), with_payload=False, with_vectors=False
        )
        found.update(str(p.id) for p in points)
    return found


def dense_store(client: QdrantClient, collection: str, embeddings: CachedEmbeddings) -> QdrantVectorStore:
    """LangChain vector store over an existing dense collection."""
    return QdrantVectorStore(client=client, collection_name=collection, embedding=embeddings)


def upsert_new_chunks(
    store: QdrantVectorStore,
    client: QdrantClient,
    collection: str,
    chunks: list[Document],
    ids: list[str],
    *,
    batch_size: int = 1024,
) -> int:
    """Embed and upsert only the chunks whose IDs are not yet in Qdrant.

    Args:
        store:      LangChain store (embeds through our cached embeddings).
        client:     Raw Qdrant client (used for the existence check).
        collection: Target collection name.
        chunks:     Chunk documents (text + metadata).
        ids:        Deterministic IDs, aligned with `chunks`.
        batch_size: Chunks per upsert round. Each round embeds in sub-batches of
                    `settings.embedding_batch_size` texts per API request.

    Returns:
        Number of chunks newly written.
    """
    present = existing_ids(client, collection, ids)
    todo = [(c, i) for c, i in zip(chunks, ids) if i not in present]
    logger.info("%d chunks total, %d already in %s, %d to add.", len(chunks), len(present), collection, len(todo))

    for start in tqdm(range(0, len(todo), batch_size), desc=f"upsert {collection}", disable=not todo):
        part = todo[start : start + batch_size]
        store.add_texts(
            texts=[c.page_content for c, _ in part],
            metadatas=[c.metadata for c, _ in part],
            ids=[i for _, i in part],
            batch_size=settings.embedding_batch_size,
        )
    return len(todo)


def doc_ids_from(results: Sequence[Document] | Sequence[tuple[Document, Any]]) -> list[str]:
    """Ranked, de-duplicated doc_ids from search results (chunks -> documents).

    The benchmark scores documents, not chunks: three chunks of one document
    count as one retrieved document, at the rank of its best chunk.
    """
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        doc = r[0] if isinstance(r, tuple) else r
        d = doc.metadata.get("doc_id")
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out
