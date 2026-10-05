#!/usr/bin/env python3
"""Make mini-redwood harder: add ~650 BM25 "hard negative" distractors from the full corpus.

Why?
    mini-redwood has ~100x fewer documents than the real corpus, so it has far
    fewer look-alike documents competing with the gold ones (Phase 1 recall: 77%
    on mini vs 46% for the paper's full-corpus vector baseline). Random noise does
    not fix that: a random Slack message rarely looks like the answer to any
    question. A *hard negative* does -- it shares the question's codenames,
    ticket IDs and jargon but is not the gold document. Those are exactly what
    trips up retrieval on the full corpus.

What gets added:
    * distractor           For every question with gold docs, the top BM25 hits
                           (question text as query) that are NOT gold for that
                           question and not already in mini-redwood:
                             1 per question  - basic, semantic, intra_document_reasoning, miscellaneous
                             2 per question  - constrained, project_related
                             3 per question  - conflicting_info, completeness
                           (the last four types are built around confusable docs).
    * high_level_evidence  High Level questions have no gold docs and their
                           supporting facts were mostly missing from the subset.
                           For each of their `answer_facts` we add the top-2 BM25
                           hits, which makes them answerable. The QA file is NOT
                           changed: these docs are not added to expected_doc_ids.

How BM25 is computed here (and why not a BM25 library):
    Indexing all ~512k documents with a library keeps every token of every
    document in memory (several GB). We only ever score ~530 queries, so we
    only need statistics for the words that appear in those queries:
      1. Build the query vocabulary (question texts + high-level facts).
      2. One parallel pass over the corpus: tokenize each file, record its
         length and the counts of query-vocabulary words only.
      3. Score each query with the standard BM25 formula in numpy.
    The pass takes a few minutes on a laptop and is cached under
    `.cache/bm25/`, so re-runs with the same questions are instant.

Cost: $0. No paid API is called. Only the ~650 new documents need embedding
later (about $0.02 when Phase 1 re-ingests).

Usage (after scripts/curate_mini_redwood.py):
    python scripts/add_distractors.py --dry-run    # show what would be added
    python scripts/add_distractors.py              # rewrite mini_redwood_docs.jsonl

Re-running is safe: previous distractor / evidence rows are dropped and
recomputed, gold and noise rows are kept as they are.

Reads:  all_documents/ (full corpus), data/raw_onyx_subset/mini_redwood_{docs,qa}.jsonl
Writes: data/raw_onyx_subset/mini_redwood_docs.jsonl (adds rows + `selection`,
        `distractor_for` columns), .cache/bm25/<vocab-hash>.npz
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from array import array
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from curate_mini_redwood import build_index, read_document, write_jsonl  # noqa: E402

from shared_utils.config import settings, setup_logging  # noqa: E402

logger = logging.getLogger("add_distractors")

# Distractors per question, by type. The confusable-by-design types get more.
PER_TYPE = {
    "basic": 1, "semantic": 1, "intra_document_reasoning": 1, "miscellaneous": 1,
    "constrained": 2, "project_related": 2,
    "conflicting_info": 3, "completeness": 3,
}
EVIDENCE_PER_FACT = 2

# Standard BM25 parameters (Robertson & Zaragoza; the defaults most engines use).
K1, B = 1.5, 0.75

# A small English stopword list. Stopwords carry almost no retrieval signal and
# dropping them keeps the query vocabulary (and the cached index) small.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we
were what when where which while who whom why will with would you your yours yourself yourselves
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, minus stopwords and 1-character tokens.

    Deliberately simple and exact (no stemming): codenames like "perf-canary"
    become "perf", "canary", and ticket IDs like "INT-7832" become "int", "7832",
    which is precisely the lexical overlap that makes a document a hard negative.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


# =============================================================================
# Corpus pass (parallel)
# =============================================================================
_VOCAB: dict[str, int] = {}
_ROOT: Path = Path()


def _init_worker(vocab: dict[str, int], root: str) -> None:
    """Give each worker process the query vocabulary once (not per task)."""
    global _VOCAB, _ROOT
    _VOCAB, _ROOT = vocab, Path(root)


def _scan(rel_path: str) -> tuple[int, list[int], list[int]]:
    """Tokenize one file. Returns (doc length, query-term ids, their counts)."""
    try:
        text = (_ROOT / rel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, [], []
    tokens = tokenize(text)
    counts = Counter(t for t in tokens if t in _VOCAB)
    return len(tokens), [_VOCAB[t] for t in counts], list(counts.values())


def build_postings(paths: list[str], vocab: dict[str, int], root: Path, cache: Path) -> dict[str, np.ndarray]:
    """Scan the corpus once and build BM25 postings for the query vocabulary.

    Returns a dict of numpy arrays:
        doc_len  (N,)        token count per document
        offsets  (V+1,)      postings of term t live in [offsets[t], offsets[t+1])
        docs     (P,)        document indices
        tfs      (P,)        term frequencies
    Cached to `cache` (keyed by the vocabulary and corpus file list).
    """
    if cache.is_file():
        logger.info("Loading cached BM25 postings from %s", cache)
        return dict(np.load(cache))

    n_vocab = len(vocab)
    # array('i') stores 4-byte ints compactly; Python lists of ints would need ~7x the memory.
    post_docs = [array("i") for _ in range(n_vocab)]
    post_tfs = [array("i") for _ in range(n_vocab)]
    doc_len = np.zeros(len(paths), dtype=np.int32)

    workers = max(1, (os.cpu_count() or 2) - 1)
    logger.info("Scanning %d documents with %d workers (one-off, cached afterwards)...", len(paths), workers)
    with Pool(workers, initializer=_init_worker, initargs=(vocab, str(root))) as pool:
        for i, (length, term_ids, counts) in enumerate(pool.imap(_scan, paths, chunksize=500)):
            doc_len[i] = length
            for t, c in zip(term_ids, counts):
                post_docs[t].append(i)
                post_tfs[t].append(c)
            if i and i % 100_000 == 0:
                logger.info("  scanned %d / %d", i, len(paths))

    offsets = np.zeros(n_vocab + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(p) for p in post_docs])
    out = {
        "doc_len": doc_len,
        "offsets": offsets,
        "docs": np.concatenate([np.frombuffer(p, dtype=np.int32) for p in post_docs]) if offsets[-1] else np.zeros(0, np.int32),
        "tfs": np.concatenate([np.frombuffer(p, dtype=np.int32) for p in post_tfs]) if offsets[-1] else np.zeros(0, np.int32),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    logger.info("Cached postings (%d entries) to %s", offsets[-1], cache)
    return out


class BM25:
    """Okapi BM25 over precomputed postings for a fixed query vocabulary."""

    def __init__(self, postings: dict[str, np.ndarray], vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.doc_len = postings["doc_len"].astype(np.float32)
        self.offsets, self.docs, self.tfs = postings["offsets"], postings["docs"], postings["tfs"]
        self.n_docs = len(self.doc_len)
        self.avgdl = float(self.doc_len.mean())
        df = np.diff(self.offsets).astype(np.float64)
        # BM25 idf with +1 inside the log so it is never negative (Lucene variant).
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def top(self, query: str, k: int) -> list[int]:
        """Indices of the `k` highest-scoring documents for `query`."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(tokenize(query)):
            t = self.vocab.get(term)
            if t is None:
                continue
            lo, hi = self.offsets[t], self.offsets[t + 1]
            d, tf = self.docs[lo:hi], self.tfs[lo:hi].astype(np.float32)
            norm = K1 * (1 - B + B * self.doc_len[d] / self.avgdl)
            scores[d] += self.idf[t] * tf * (K1 + 1) / (tf + norm)
        k = min(k, self.n_docs)
        cand = np.argpartition(-scores, k)[:k]
        cand = cand[scores[cand] > 0]
        return cand[np.argsort(-scores[cand])].tolist()


# =============================================================================
# Selection
# =============================================================================
def select(qa: pd.DataFrame, index: pd.DataFrame, bm25: BM25, existing_paths: set[str]) -> pd.DataFrame:
    """Pick distractors and high-level evidence. Returns index rows + selection metadata."""
    taken: dict[int, dict] = {}  # corpus row -> {"selection", "distractor_for"}
    paths = index["path"].tolist()
    doc_ids = index["doc_id"].tolist()

    def take(row: int, selection: str, qid: str) -> None:
        if row in taken:
            # Already picked for another question: just record that it matters here too.
            taken[row]["distractor_for"].append(qid)
        else:
            taken[row] = {"selection": selection, "distractor_for": [qid]}

    for q in qa.itertuples(index=False):
        gold = set(q.expected_doc_ids)
        if q.question_type == "high_level":
            for fact in q.answer_facts:
                added = 0
                for row in bm25.top(fact, 20):
                    if paths[row] in existing_paths:
                        continue  # evidence already present; nothing to add
                    take(row, "high_level_evidence", q.question_id)
                    added += 1
                    if added == EVIDENCE_PER_FACT:
                        break
            continue

        want = PER_TYPE.get(q.question_type, 0)
        if not want or not gold:
            continue  # info_not_found: nothing to distract from
        got = 0
        for row in bm25.top(q.question, 50):
            if doc_ids[row] in gold or paths[row] in existing_paths:
                continue
            if row in taken:
                # Shared distractor: counts for this question but adds no new document.
                take(row, taken[row]["selection"], q.question_id)
                got += 1
            else:
                take(row, "distractor", q.question_id)
                got += 1
            if got == want:
                break

    rows = index.loc[list(taken)].copy()
    rows["selection"] = [taken[r]["selection"] for r in taken]
    rows["distractor_for"] = [taken[r]["distractor_for"] for r in taken]
    rows["is_gold"] = False
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", type=Path, default=settings.paths.corpus_dir)
    p.add_argument("--seed", type=int, default=settings.seed, help="Seed for the final shuffle.")
    p.add_argument("--dry-run", action="store_true", help="Show the selection; write nothing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        docs_path = settings.paths.mini_docs
        docs = pd.read_json(docs_path, lines=True)
        qa = pd.read_json(settings.paths.mini_qa, lines=True)
        if "selection" not in docs.columns:  # files from before this column existed
            docs["selection"] = np.where(docs["is_gold"], "gold", "noise")
        base = docs[docs["selection"].isin(["gold", "noise"])].drop(columns=["distractor_for"], errors="ignore")
        if len(base) < len(docs):
            logger.info("Dropping %d previously added rows; recomputing.", len(docs) - len(base))

        # 1. Query vocabulary: every token any query will use.
        queries = list(qa["question"]) + [f for facts in qa.loc[qa["question_type"] == "high_level", "answer_facts"] for f in facts]
        vocab_terms = sorted({t for q in queries for t in tokenize(q)})
        vocab = {t: i for i, t in enumerate(vocab_terms)}
        logger.info("%d queries -> query vocabulary of %d terms.", len(queries), len(vocab))

        # 2. Corpus pass (cached by vocabulary + corpus listing).
        index = build_index(args.source_dir)
        key = hashlib.sha256(("\n".join(vocab_terms) + "\n" + "\n".join(index["path"])).encode()).hexdigest()[:16]
        postings = build_postings(index["path"].tolist(), vocab, args.source_dir,
                                  settings.paths.root / ".cache" / "bm25" / f"{key}.npz")
        bm25 = BM25(postings, vocab)

        # 3. Select.
        added = select(qa, index, bm25, set(base["path"]))
        summary = added.groupby(["selection", "source_type"]).size().unstack(fill_value=0)
        logger.info("Selected %d new documents:\n%s", len(added), summary.to_string())
        shared = int((added["distractor_for"].str.len() > 1).sum())
        logger.info("%d distractors are shared by more than one question.", shared)

        if args.dry_run:
            logger.info("--dry-run: nothing written.")
            return 0

        # 4. Read the new files, merge, shuffle, write.
        titles, contents = zip(*(read_document(args.source_dir, p) for p in added["path"]))
        added["title"], added["content"] = list(titles), list(contents)
        base = base.assign(distractor_for=None)
        cols = ["doc_id", "source_type", "title", "content", "path", "is_gold", "selection", "distractor_for"]
        merged = pd.concat([base[cols], added[cols]], ignore_index=True)
        merged = merged.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        write_jsonl(merged, docs_path)
        logger.info("mini-redwood now has %d documents:\n%s", len(merged), merged["selection"].value_counts().to_string())
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
