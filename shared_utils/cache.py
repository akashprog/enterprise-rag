"""A tiny on-disk cache so that no paid API call is ever made twice.

Why this exists (cost):
    A RAG experiment loop re-runs the same things constantly: re-evaluate after
    a prompt tweak, re-ingest after a crash, re-score a phase to build a README
    table. Without a cache every re-run re-pays for identical embeddings,
    identical judge verdicts and identical answers. With it, a re-run only pays
    for what actually changed.

How it works:
    * Every cacheable call is reduced to a deterministic *key*: a SHA-256 hash
      of everything that can change the output (call kind, model name, prompt
      version, the exact input, generation parameters). If any of those change,
      the key changes and we pay for a fresh call -- stale results are never
      served for a different request.
    * Values are stored in SQLite, which is a single local file, needs no
      server, survives crashes, and handles hundreds of thousands of rows.
    * Two tables:
        - `json_cache`   : arbitrary JSON results (chat answers, judge verdicts,
                           and from Step 4, Jev answers).
        - `vector_cache` : embeddings stored as raw float32 bytes (~4x smaller
                           and much faster to load than JSON lists of floats).

Invalidation:
    Bump the `prompt_version` you pass in when you change a prompt's wording.
    To wipe everything, delete the file at `settings.paths.cache_db`.

Reads/writes: `.cache/api_cache.sqlite` (path from shared_utils.config).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from shared_utils.config import settings


def make_key(*parts: Any) -> str:
    """Build a deterministic cache key from any JSON-serialisable parts.

    `sort_keys=True` makes dicts hash identically regardless of insertion order,
    so `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` share one cache entry.

    Args:
        *parts: Anything that influences the API output (kind, model, prompt
            version, messages, parameters...).

    Returns:
        A 64-character hex SHA-256 digest.
    """
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ApiCache:
    """Thread-safe SQLite cache for JSON results and embedding vectors.

    One instance is shared per process (see `get_cache()`). All methods are
    synchronous and fast (local disk); they are safe to call from asyncio code
    because each call holds the lock only for a single short SQL statement.

    Args:
        db_path: SQLite file location. Parent directories are created.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets asyncio executors / worker threads share
        # the connection; `self._lock` serialises access so that is safe.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            # WAL mode: readers never block the writer, and a crash mid-run
            # cannot corrupt entries that were already committed.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS json_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vector_cache (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
            )
            self._conn.commit()

    # ------------------------------------------------------------ JSON values
    def get_json(self, key: str) -> Any | None:
        """Return the cached JSON value for `key`, or None on a miss."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM json_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_json(self, key: str, value: Any) -> None:
        """Store `value` (must be JSON-serialisable) under `key`, overwriting."""
        encoded = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO json_cache (key, value) VALUES (?, ?)", (key, encoded)
            )
            # Commit per write: a crash loses at most the in-flight call, never
            # results we already paid for.
            self._conn.commit()

    # ---------------------------------------------------------------- vectors
    def get_vectors(self, keys: list[str]) -> dict[str, list[float]]:
        """Bulk-fetch embeddings. Returns only the keys that were found.

        Bulk lookup matters at ingest time: checking 50k chunk keys one by one
        would be 50k round-trips; here it is a handful of `IN (...)` queries.
        """
        found: dict[str, list[float]] = {}
        # SQLite limits the number of `?` placeholders per statement; 900 stays
        # safely under the historical default limit of 999.
        for start in range(0, len(keys), 900):
            chunk = keys[start : start + 900]
            placeholders = ",".join("?" * len(chunk))
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT key, vec FROM vector_cache WHERE key IN ({placeholders})", chunk
                ).fetchall()
            for key, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32).tolist()
        return found

    def set_vectors(self, items: Iterable[tuple[str, list[float]]]) -> None:
        """Bulk-store embeddings as float32 bytes in a single transaction."""
        rows = [(key, np.asarray(vec, dtype=np.float32).tobytes()) for key, vec in items]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vector_cache (key, vec) VALUES (?, ?)", rows
            )
            self._conn.commit()

    # ------------------------------------------------------------------ misc
    def stats(self) -> dict[str, int]:
        """Row counts per table -- handy for "how much have we cached?" checks."""
        with self._lock:
            n_json = self._conn.execute("SELECT COUNT(*) FROM json_cache").fetchone()[0]
            n_vec = self._conn.execute("SELECT COUNT(*) FROM vector_cache").fetchone()[0]
        return {"json_entries": n_json, "vector_entries": n_vec}


_cache: ApiCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> ApiCache:
    """Return the process-wide `ApiCache`, creating it on first use.

    Lazy creation means importing this module never touches the disk; tools
    that don't call paid APIs (e.g. curation) never create the cache file.
    """
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = ApiCache(settings.paths.cache_db)
        return _cache
