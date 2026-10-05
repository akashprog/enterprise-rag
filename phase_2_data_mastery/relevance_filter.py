"""Phase 2, Step 2.2: keep a dynamic number of arm-A passages, chosen for relevance.

The one change against arm A (small-to-big, N=5, window ±1, 2,000-token whole-document
threshold, `results/phase_2/n5_w1_t2000_b8000/`):
    Matching is the same cosine search over `p1_naive`. Each candidate document is
    turned into the same passage arm A would send (`passage_for`: the whole document
    when it is short, otherwise the matched chunks ±1 neighbour). The Phase 1 prompt
    is unchanged. Only WHICH documents are selected changes, and how many.

Why N=5 is the wrong constant
    Arm A beat an equal-size dump of plain chunks, but it lost recall (69.6 vs
    75.1) because some questions need more than 5 documents and some need fewer.
    A filter that reads each candidate passage and keeps the useful ones can spend
    the token budget on the documents that actually help.

Two recalls, because they answer different questions
    * Set recall: fraction of the gold documents present anywhere in the kept set.
      This is the ceiling a filter can reach: a document never retrieved cannot be
      selected. Reported for the top 10/20/30/50 candidates.
    * Recall@10: the benchmark metric, which only credits gold among the first 10
      kept documents. Once a run keeps more than 10, the two diverge. Invalid extra
      docs counts every non-gold document kept, as everywhere else in the series.
    Question types with no gold documents (high_level, info_not_found) are excluded
    from both, matching the benchmark.

Cost
    Part 1 (candidate recall, the oracle, score cutoffs, the miss breakdown) is
    free: cached embeddings, local Qdrant, no model calls.
    Part 2 asks Jev, in one request per question, whether each candidate passage
    helps. Jev reads the state once and answers every yes/no in that request
    together. A question whose passages do not fit the context budget is split
    into as few requests as needed. Jev bills input tokens only ($0.042 / 1M).
    Every call is cached, so the threshold sweep and any re-run are free.
    --dry-run prices the calls without making them.

Usage:
    python -m phase_2_data_mastery.relevance_filter --free
    python -m phase_2_data_mastery.relevance_filter --dry-run
    python -m phase_2_data_mastery.relevance_filter --limit 20          # smoke test
    python -m phase_2_data_mastery.relevance_filter                     # score all (v2), then sweep
    python -m phase_2_data_mastery.relevance_filter --sweep             # sweep the v1 scores; no calls

Reads:  Qdrant `p1_naive`, data/raw_onyx_subset/mini_redwood_{docs,qa}.jsonl
Writes: results/phase_2/relevance_filter/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_2_data_mastery.query_small_to_big import (  # noqa: E402
    DEFAULT_WHOLE_DOC_MAX,
    DEFAULT_WINDOW,
    MIN_PARTIAL_TOKENS,
    DocIndex,
    messages,
    passage_for,
    phase1_output_tokens_per_answer,
)
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import dedupe, load_qa, retrieval_metrics  # noqa: E402
from shared_utils.jev_judge import CachedJev  # noqa: E402
from shared_utils.llm import CachedChat, count_tokens, run_limited, truncate_to_tokens, usage  # noqa: E402
from shared_utils.runner import run_phase  # noqa: E402

logger = logging.getLogger("phase2.relevance_filter")

# Chunks retrieved per question. Measured on this corpus: the median question
# needs 69 chunks to surface 50 distinct documents, so 200 leaves margin.
SEARCH_K = 200
# How many of the best documents a filter is allowed to choose from.
CANDIDATES = 30
# Set-recall ceilings to report. 30 is the filter's pool; 50 is as deep as we look.
CEILINGS = (10, 20, 30, 50)
# Score-cutoff rules tried in Part 1, all applied to the same top-30 pool.
# Chosen from the measured curve: a document's score is about 0.97 of the best at
# rank 2 and about 0.88 at rank 10, falling ~0.02 per rank, so a fixed gap of 0.05
# (or a ratio of 0.80) keeps nearly the whole pool and is not a cutoff.
RELATIVE_CUTOFFS = (0.95, 0.92, 0.90)
# Jev yes/no question. The version is part of the cache key.
# v1 was one request per passage. v2 puts every passage of a question into one
# request (split only when they would exceed Jev's context budget). The two do
# not share a cache: each v2 question can see the other passages in the state.
RELEVANCE_PROMPT_VERSION_V1 = "jev-relevance-v1"
RELEVANCE_PROMPT_VERSION = "jev-relevance-v2"
# jev-1.13.0 allows 32k tokens for the state plus the longest question, and 64k
# for the state plus every question. Local token counts ran about 1.22x under
# the bill on the v1 scoring, so a call is packed to 24k local tokens
# (24k * 1.22 = 29k, under the 32k ceiling).
LOCAL_CALL_BUDGET = 24_000
# Account limits: 100k tokens/s and 40 requests/s. Pace under both.
JEV_TOKENS_PER_SECOND = 80_000
BILLED_TOKEN_RATIO = 1.22
THRESHOLDS = (0.3, 0.5, 0.7)
FLOORS = (0, 1)
DEFAULT_BUDGET = 10_000
# Stay under Jev's 40 requests/s limit.
JEV_PER_SECOND = 30

OUT_DIR = settings.paths.results_dir / "phase_2" / "relevance_filter"
CANDIDATES_PATH = OUT_DIR / "candidates.jsonl"
# v1 probabilities, one request per passage. The answered Phase 2.2 run and
# `--checks` read this file. A v2 scoring run must not overwrite it.
SCORES_PATH = OUT_DIR / "jev_scores.jsonl"
SCORES_PATH_V2 = OUT_DIR / "jev_scores_v2.jsonl"


def _helps_noul(passage_name: str) -> dict[str, Any]:
    """One yes/no: does this named passage help answer `question`?

    The instruction names the state field because Jev reads literally. The
    criteria spell out the boundary between "bears on" and "merely nearby".
    """
    return {
        "type": "noul",
        "instructions": f"Does `{passage_name}` contain information that helps answer `question`?",
        "criteria": {
            "true": f"`{passage_name}` states facts, names, numbers, dates or decisions that bear on `question`",
            "false": f"`{passage_name}` is about a different topic, or too generic to help answer `question`",
        },
    }


def relevance_questions() -> dict[str, dict[str, Any]]:
    """The v1 question, one passage per request. Used only to read that cache."""
    return {"helps": _helps_noul("passage")}


def _call_tokens(state: dict[str, Any], questions: dict[str, Any]) -> int:
    """Local token count of one Jev request. Same strings `CachedJev` prices."""
    return count_tokens(json.dumps(state) + json.dumps(questions))


def _one_batch(question: str, passages: list[str]) -> dict[str, Any]:
    """One request: the question plus these passages, and a yes/no for each."""
    state: dict[str, Any] = {"question": question}
    spec: dict[str, Any] = {}
    for i, text in enumerate(passages):
        name = f"passage_{i}"
        state[name] = text
        spec[f"p{i}"] = _helps_noul(name)
    return {"state": state, "questions": spec, "n": len(passages),
            "local_tokens": _call_tokens(state, spec)}


def relevance_batches(question: str, passages: list[str]) -> list[dict[str, Any]]:
    """Pack passages into as few requests as fit under `LOCAL_CALL_BUDGET`.

    Order is preserved, so answer `p0` of a batch is the first passage in it.
    A passage that is already over the budget is sent alone.
    """
    batches: list[dict[str, Any]] = []
    start = 0
    while start < len(passages):
        end = start + 1
        while end < len(passages):
            if _one_batch(question, passages[start:end + 1])["local_tokens"] > LOCAL_CALL_BUDGET:
                break
            end += 1
        batches.append(_one_batch(question, passages[start:end]))
        start = end
    return batches


# =============================================================================
# Candidates: one cosine search, grouped into documents
# =============================================================================
@dataclass
class Candidate:
    """One document surfaced by the chunk search, best chunk first."""

    doc_id: str
    path: str
    score: float  # cosine similarity of its best chunk; higher is better
    chunks: list[int]  # chunk indexes in that document, in retrieval order
    source_type: str
    passage_tokens: int = 0
    mode: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "path": self.path, "score": round(self.score, 4),
                "chunks": self.chunks, "source_type": self.source_type,
                "passage_tokens": self.passage_tokens, "mode": self.mode}


@dataclass
class QuestionCandidates:
    question_id: str
    question_type: str
    question: str
    gold: list[str]
    candidates: list[Candidate] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"question_id": self.question_id, "question_type": self.question_type,
                "question": self.question, "gold": self.gold,
                "candidates": [c.to_json() for c in self.candidates]}


def search_candidates(question: str, store, index: DocIndex, *, k: int = SEARCH_K) -> list[Candidate]:
    """Top documents for one question: best chunk first, one entry per document.

    Documents are ordered by their highest-scoring chunk. Later chunks of a
    document are recorded too, because the passage window is built around every
    chunk of it that the search returned. Two files that share a doc_id collapse
    to the higher-scoring one, since the benchmark scores doc_ids.
    """
    by_path: dict[str, Candidate] = {}
    order: list[str] = []
    for doc, score in store.similarity_search_with_score(question, k=k):
        path = doc.metadata["path"]
        if path not in by_path:
            split = index.get(path)
            order.append(path)
            by_path[path] = Candidate(split.doc_id, path, float(score), [], index.source_of_path(path))
        cand = by_path[path]
        cand.chunks.append(int(doc.metadata["chunk_index"]))
        cand.score = max(cand.score, float(score))

    # First occurrence is already the best chunk, because the search is ranked.
    collapsed: dict[str, Candidate] = {}
    ranked: list[Candidate] = []
    for path in order:
        cand = by_path[path]
        kept = collapsed.get(cand.doc_id)
        if kept is None:
            collapsed[cand.doc_id] = cand
            ranked.append(cand)
        elif cand.score > kept.score:
            ranked[ranked.index(kept)] = cand
            collapsed[cand.doc_id] = cand
    return ranked


def attach_passages(cands: list[Candidate], index: DocIndex) -> None:
    """Fill each candidate's passage shape (arm A's rule, no token budget yet)."""
    for c in cands:
        c.mode, _, c.passage_tokens = passage_for(
            index.get(c.path), c.chunks, window=DEFAULT_WINDOW, whole_doc_max=DEFAULT_WHOLE_DOC_MAX)


def passage_text(c: Candidate, index: DocIndex) -> str:
    """Rebuild the passage text. Deterministic, so it is not stored on disk."""
    _, text, _ = passage_for(index.get(c.path), c.chunks, window=DEFAULT_WINDOW,
                             whole_doc_max=DEFAULT_WHOLE_DOC_MAX)
    return text


def build_all(store, index: DocIndex, qa) -> list[QuestionCandidates]:
    """Search every question. Free: embeddings are cached and Qdrant is local."""
    out: list[QuestionCandidates] = []
    for row in qa.itertuples():
        qc = QuestionCandidates(row.question_id, row.question_type, row.question, list(row.expected_doc_ids))
        qc.candidates = search_candidates(row.question, store, index)[: max(CEILINGS)]
        attach_passages(qc.candidates, index)
        out.append(qc)
    return out


def save_candidates(questions: list[QuestionCandidates]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_PATH.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_json(), ensure_ascii=False) + "\n")


def load_candidates() -> list[QuestionCandidates]:
    """Read the candidate file written by the free check."""
    if not CANDIDATES_PATH.is_file():
        raise FileNotFoundError(f"{CANDIDATES_PATH} not found. Run --free first.")
    out = []
    for line in CANDIDATES_PATH.open(encoding="utf-8"):
        raw = json.loads(line)
        qc = QuestionCandidates(raw["question_id"], raw["question_type"], raw["question"], raw["gold"])
        qc.candidates = [Candidate(**{k: c[k] for k in ("doc_id", "path", "score", "chunks", "source_type",
                                                        "passage_tokens", "mode")}) for c in raw["candidates"]]
        out.append(qc)
    return out


# =============================================================================
# Selection rules
# =============================================================================
def score_cutoff(cands: list[Candidate], *, ratio: float | None = None, elbow: bool = False) -> list[Candidate]:
    """Keep a prefix of `cands` (already best-first) by a similarity rule.

    `ratio`: keep while the score stays at least this fraction of the best score.
    `elbow`: keep up to and including the document just before the single largest
             drop in the list (the sharpest fall-off).
    """
    if not cands:
        return []
    if elbow:
        if len(cands) < 2:
            return cands
        gaps = [a.score - b.score for a, b in zip(cands, cands[1:])]
        return cands[: gaps.index(max(gaps)) + 1]
    kept = [cands[0]]
    for cur in cands[1:]:
        if cur.score < ratio * cands[0].score:
            break
        kept.append(cur)
    return kept


def select_by_probability(cands: list[Candidate], probs: list[float], *, threshold: float,
                          budget: int, floor: int) -> list[tuple[Candidate, str, int]]:
    """Documents to send: probability >= threshold, best-first, within `budget` tokens.

    `floor` forces the first documents in rank order to be kept even when they
    fall below the threshold, so a floor of 1 never sends an empty context.
    The last document is cut to fill the budget, exactly as arm A does, and a
    cut smaller than MIN_PARTIAL_TOKENS is left out instead.

    Returns:
        (candidate, mode, tokens) for each document actually sent. The mode gains
        a "+cut" suffix when the passage was truncated to fit.
    """
    chosen = [i for i, p in enumerate(probs) if p >= threshold]
    if len(chosen) < floor:
        for i in range(len(cands)):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= floor:
                break
    chosen.sort()

    sent: list[tuple[Candidate, str, int]] = []
    used = 0
    for i in chosen:
        cand = cands[i]
        mode, n = cand.mode, cand.passage_tokens
        remaining = budget - used
        if n > remaining:
            if remaining < MIN_PARTIAL_TOKENS:
                break
            mode, n = mode + "+cut", remaining
        sent.append((cand, mode, n))
        used += n
        if used >= budget or mode.endswith("+cut"):
            break
    return sent


def prompt_tokens(question: str, passages: list[tuple[str, int]]) -> int:
    """Tokens of the Phase 1 prompt around these passages.

    `passages` is (text, tokens). Token counts are summed rather than re-encoded
    as one string; the join between passages is a blank line, worth ~2 tokens,
    which is included.
    """
    if not passages:
        body = ""
    else:
        body = "\n\n".join(text for text, _ in passages)
    return sum(count_tokens(t) for _, t in messages(question, body))


def prompt_tokens_from_counts(question: str, token_counts: list[int]) -> int:
    """Same total, using precomputed passage sizes plus the empty-prompt overhead.

    The overhead is the Phase 1 prompt with an empty context, counted once per
    question; each extra passage also costs the blank line that joins it.
    """
    overhead = sum(count_tokens(t) for _, t in messages(question, ""))
    joins = 2 * max(0, len(token_counts) - 1)
    return overhead + sum(token_counts) + joins


# =============================================================================
# Metrics
# =============================================================================
def set_recall(kept: list[str], gold: list[str]) -> float | None:
    """Fraction of gold documents present anywhere in `kept` (the filter's ceiling)."""
    gold_set = set(gold)
    if not gold_set:
        return None
    # Same denominator as retrieval_metrics: repeated gold ids count once.
    return len(set(kept) & gold_set) / len(gold_set)


def summarise_kept(questions: list[QuestionCandidates], kept_ids: dict[str, list[str]]) -> dict[str, Any]:
    """Recall@10, set recall, invalid extras and document counts for one selection."""
    rows = []
    for q in questions:
        ids = kept_ids[q.question_id]
        recall10, invalid = retrieval_metrics(ids, q.gold, settings.recall_k)
        rows.append({"type": q.question_type, "n": len(ids), "recall10": recall10,
                     "set_recall": set_recall(ids, q.gold), "invalid": invalid,
                     "zero": len(ids) == 0})
    return _aggregate(rows)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall and per-question-type means. Recall ignores types without gold."""

    def pack(group: list[dict[str, Any]]) -> dict[str, Any]:
        def mean(key: str, scale: float = 1.0) -> float | None:
            vals = [r[key] for r in group if r[key] is not None]
            return None if not vals else round(scale * sum(vals) / len(vals), 2)

        counts = [r["n"] for r in group]
        return {
            "n": len(group),
            "recall10": mean("recall10", 100),
            "set_recall": mean("set_recall", 100),
            "invalid": mean("invalid"),
            "avg_docs": round(statistics.mean(counts), 2) if counts else None,
            "docs_median": statistics.median(counts) if counts else None,
            "docs_p90": sorted(counts)[int(0.9 * (len(counts) - 1))] if counts else None,
            "docs_max": max(counts) if counts else None,
            "zero_docs_pct": round(100 * sum(r["zero"] for r in group) / len(group), 1) if group else None,
        }

    by_type = {t: pack([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}
    return {"overall": pack(rows), "by_type": by_type}


def estimate_cost(total_prompt_tokens: int, n_questions: int) -> float | None:
    """Answering cost, assuming Phase 1's average output length."""
    price = settings.model_prices.get(settings.answer_model)
    out_per = phase1_output_tokens_per_answer()
    if price is None or out_per is None:
        return None
    return round(total_prompt_tokens * price.input / 1e6 + n_questions * out_per * price.output / 1e6, 4)


# =============================================================================
# Part 1: free checks
# =============================================================================
def free_checks(questions: list[QuestionCandidates]) -> dict[str, Any]:
    """Candidate ceilings, the oracle, three score cutoffs, and where the misses are."""
    pool = {q.question_id: q.candidates[:CANDIDATES] for q in questions}

    ceilings = {}
    for n in CEILINGS:
        kept = {q.question_id: [c.doc_id for c in q.candidates[:n]] for q in questions}
        ceilings[f"top_{n}"] = summarise_kept(questions, kept)
        short = sum(len(q.candidates) < n for q in questions)
        ceilings[f"top_{n}"]["questions_with_fewer_candidates"] = short

    # Oracle: only the gold documents inside the top-30 pool, in the order retrieved.
    oracle_kept = {}
    for q in questions:
        gold = set(q.gold)
        oracle_kept[q.question_id] = [c.doc_id for c in q.candidates[:CANDIDATES] if c.doc_id in gold]

    cutoffs = {
        f"relative_{ratio}": {q.question_id: [c.doc_id for c in score_cutoff(pool[q.question_id], ratio=ratio)]
                              for q in questions}
        for ratio in RELATIVE_CUTOFFS
    }
    cutoffs["elbow"] = {q.question_id: [c.doc_id for c in score_cutoff(pool[q.question_id], elbow=True)]
                        for q in questions}

    # Gold documents the top-30 pool never sees, by the document's source type.
    missed: Counter[str] = Counter()
    missed_questions = 0
    gold_total = 0
    for q in questions:
        if not q.gold:
            continue
        found = {c.doc_id for c in q.candidates[:CANDIDATES]}
        miss = [d for d in dedupe(q.gold) if d not in found]
        gold_total += len(dedupe(q.gold))
        if miss:
            missed_questions += 1
            for doc_id in miss:
                missed[source_of_gold(doc_id)] += 1

    return {
        "search_k": SEARCH_K,
        "ceilings": ceilings,
        "oracle_top30": summarise_kept(questions, oracle_kept),
        "score_cutoffs": {name: summarise_kept(questions, kept) for name, kept in cutoffs.items()},
        "misses": {
            "questions_with_a_gold_doc_outside_top30": missed_questions,
            "questions_with_gold": sum(bool(q.gold) for q in questions),
            "gold_docs_missed": sum(missed.values()),
            "gold_docs_total": gold_total,
            "missed_by_source": dict(missed.most_common()),
        },
        "passage_tokens": _token_stats([c.passage_tokens for q in questions for c in q.candidates[:CANDIDATES]]),
    }


_GOLD_SOURCE: dict[str, str] = {}


def source_of_gold(doc_id: str) -> str:
    """Source type of a gold document, loaded once from mini-redwood."""
    if not _GOLD_SOURCE:
        from phase_1_baseline.ingest_naive import load_docs

        for row in load_docs(None).itertuples():
            # A handful of doc_ids are shared by two files; prefer the gold copy.
            if row.doc_id not in _GOLD_SOURCE or row.is_gold:
                _GOLD_SOURCE[row.doc_id] = row.source_type
    return _GOLD_SOURCE.get(doc_id, "unknown")


def _token_stats(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {"median": ordered[len(ordered) // 2], "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "max": ordered[-1]}


def best_cutoff(cutoffs: dict[str, dict[str, Any]]) -> str:
    """Highest Recall@10; ties go to fewer invalid documents, then fewer documents.

    Set recall favours the rule that barely filters, since keeping everything
    finds the most gold. Recall@10 is the metric the series is scored on.
    """
    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float]:
        overall = item[1]["overall"]
        return (overall["recall10"] or 0, -(overall["invalid"] or 0), -(overall["avg_docs"] or 0))

    return max(cutoffs.items(), key=key)[0]


# =============================================================================
# Part 2: Jev relevance
# =============================================================================
class _CallPacer:
    """Spaces out real Jev calls so a burst stays under the token and request caps.

    A batched call can hold ~24k local tokens. At 40 requests/s that would be
    far over the 100k tokens/s account limit, so the next call waits until both
    budgets have room. Cached calls and dry runs do not wait.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._next_request = 0.0
        self._tokens_free_at = 0.0

    async def wait(self, local_tokens: int) -> None:
        billed = local_tokens * BILLED_TOKEN_RATIO
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_request, self._tokens_free_at)
            self._next_request = start + 1.0 / JEV_PER_SECOND
            self._tokens_free_at = start + billed / JEV_TOKENS_PER_SECOND
            delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)


async def score_passages(questions: list[QuestionCandidates], *, dry_run: bool) -> dict[str, list[float]]:
    """Jev's P(passage helps) for the top `CANDIDATES` documents of each question.

    One request per question where the passages fit, otherwise several, in
    candidate order. Returns question_id -> probabilities in that same order.
    Cached calls cost nothing; a dry run only counts tokens.
    """
    jev = CachedJev(dry_run=dry_run)
    index = DocIndex()
    pacer = _CallPacer()
    probs: dict[str, list[float | None]] = {}
    jobs: list[tuple[str, list[int], dict[str, Any]]] = []
    split = 0
    # Passage text is built here, in one thread. DocIndex is not safe to share
    # across the concurrent calls below.
    for q in questions:
        pool = q.candidates[:CANDIDATES]
        texts = [passage_text(c, index) for c in pool]
        batches = relevance_batches(q.question, texts)
        split += len(batches) > 1
        probs[q.question_id] = [None] * len(pool)
        offset = 0
        for batch in batches:
            ranks = list(range(offset, offset + batch["n"]))
            offset += batch["n"]
            jobs.append((q.question_id, ranks, batch))
    if split:
        logger.info("%d questions need more than one Jev call to stay under the context budget.", split)

    async def one(job: tuple[str, list[int], dict[str, Any]]) -> None:
        qid, ranks, batch = job
        state, spec = batch["state"], batch["questions"]
        if not dry_run and jev.peek(state, spec, prompt_version=RELEVANCE_PROMPT_VERSION) is None:
            await pacer.wait(batch["local_tokens"])
        for attempt in range(4):
            try:
                result = await jev.ask(state, spec, prompt_version=RELEVANCE_PROMPT_VERSION)
                break
            except Exception as exc:  # noqa: BLE001 - transient rate limits and timeouts
                if attempt == 3 or dry_run:
                    raise
                logger.warning("Jev call failed (%s), retry %d", exc, attempt + 1)
                await asyncio.sleep(2 ** attempt)
        if result is None:
            return
        for local, rank in enumerate(ranks):
            probs[qid][rank] = result["answers"][f"p{local}"]["noul"]

    try:
        await run_limited(jobs, one, limit=8, desc="jev relevance")
    finally:
        await jev.aclose()
    return {qid: [p if p is not None else float("nan") for p in ps] for qid, ps in probs.items()}


def save_scores(questions: list[QuestionCandidates], probs: dict[str, list[float]],
               path: Path = SCORES_PATH_V2) -> None:
    """One row per question, probabilities aligned with its candidate list.

    Defaults to the v2 file. The v1 file is the Phase 2.2 record.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({"question_id": q.question_id,
                                "probs": [round(p, 4) for p in probs[q.question_id]]}) + "\n")


def load_scores() -> dict[str, list[float]]:
    if not SCORES_PATH.is_file():
        raise FileNotFoundError(f"{SCORES_PATH} not found. Score the passages first.")
    return {r["question_id"]: r["probs"] for r in map(json.loads, SCORES_PATH.open(encoding="utf-8"))}


def sweep(questions: list[QuestionCandidates], probs: dict[str, list[float]], *,
          budget: int) -> dict[str, Any]:
    """Threshold x floor grid: retrieval metrics, documents kept, prompt tokens, cost."""
    grid = {}
    for threshold in THRESHOLDS:
        for floor in FLOORS:
            kept: dict[str, list[str]] = {}
            token_total = 0
            for q in questions:
                pool = q.candidates[:CANDIDATES]
                sent = select_by_probability(pool, probs[q.question_id], threshold=threshold,
                                              budget=budget, floor=floor)
                kept[q.question_id] = [c.doc_id for c, _, _ in sent]
                token_total += prompt_tokens_from_counts(q.question, [n for _, _, n in sent])
            name = f"p{threshold}_floor{floor}"
            stats = summarise_kept(questions, kept)
            stats["prompt_tokens_avg"] = round(token_total / len(questions))
            stats["est_answer_cost_usd"] = estimate_cost(token_total, len(questions))
            inf = [q for q in questions if q.question_type == "info_not_found"]
            stats["info_not_found_zero_docs_pct"] = round(
                100 * sum(len(kept[q.question_id]) == 0 for q in inf) / len(inf), 1)
            grid[name] = stats
    return grid


def probability_summary(questions: list[QuestionCandidates], probs: dict[str, list[float]]) -> dict[str, Any]:
    """Is the signal usable: spread, and whether gold passages score higher."""
    gold_ps: list[float] = []
    other_ps: list[float] = []
    for q in questions:
        gold = set(q.gold)
        for cand, p in zip(q.candidates[:CANDIDATES], probs[q.question_id]):
            (gold_ps if cand.doc_id in gold else other_ps).append(p)

    def pack(xs: list[float]) -> dict[str, float] | None:
        if not xs:
            return None
        xs = sorted(xs)
        return {"n": len(xs), "mean": round(statistics.mean(xs), 3),
                "p10": round(xs[len(xs) // 10], 3), "median": round(statistics.median(xs), 3),
                "p90": round(xs[int(0.9 * (len(xs) - 1))], 3)}

    return {"gold_passages": pack(gold_ps), "other_passages": pack(other_ps)}


# =============================================================================
# Reporting
# =============================================================================
def _fmt(stats: dict[str, Any]) -> str:
    o = stats["overall"]
    return (f"recall@10 {o['recall10']}  set-recall {o['set_recall']}  invalid {o['invalid']}  "
            f"docs avg {o['avg_docs']} median {o['docs_median']} p90 {o['docs_p90']} max {o['docs_max']}")


def print_free(report: dict[str, Any]) -> None:
    print("\n=== Candidate ceilings (set recall = gold found anywhere in the kept documents) ===")
    for n in CEILINGS:
        s = report["ceilings"][f"top_{n}"]
        print(f"top {n:2d}  {_fmt(s)}   (questions with fewer than {n} candidates: {s['questions_with_fewer_candidates']})")
    print("\n=== Oracle: gold documents only, inside the top 30 ===")
    print(_fmt(report["oracle_top30"]))
    print("\n=== Score cutoffs, applied to the top 30 ===")
    winner = best_cutoff(report["score_cutoffs"])
    for name, stats in report["score_cutoffs"].items():
        mark = "  <-- best Recall@10 of these rules" if name == winner else ""
        print(f"{name:16s} {_fmt(stats)}{mark}")
    misses = report["misses"]
    print("\n=== Gold documents outside the top 30 ===")
    print(f"{misses['questions_with_a_gold_doc_outside_top30']} of {misses['questions_with_gold']} questions "
          f"with gold miss at least one; {misses['gold_docs_missed']} of {misses['gold_docs_total']} gold documents")
    print("missed by source:", ", ".join(f"{k} {v}" for k, v in misses["missed_by_source"].items()))
    tokens = report["passage_tokens"]
    print(f"passage size in the top 30: median {tokens['median']} tokens, p90 {tokens['p90']}, max {tokens['max']}")


def arm_a_and_references(free: dict[str, Any]) -> dict[str, Any]:
    """Rows that the sweep table is compared against. Numbers come from saved runs."""
    arm = json.loads((settings.paths.results_dir / "phase_2" / "n5_w1_t2000_b8000"
                      / "jev" / "answers_metrics.json").read_text())["overall"]
    summary = json.loads((settings.paths.results_dir / "phase_2" / "contexts"
                          / "n5_w1_t2000_b8000" / "summary.json").read_text())
    # Arm A keeps 5 documents, so set recall and Recall@10 are the same number.
    arm_row = {"overall": {"n": arm["n"], "recall10": arm["recall_at_k"], "set_recall": arm["recall_at_k"],
                           "invalid": arm["invalid_extra_docs"], "avg_docs": summary["avg_docs_sent"],
                           "docs_median": None, "docs_p90": None, "docs_max": None, "zero_docs_pct": 0.0},
               "prompt_tokens_avg": summary["avg_prompt_tokens"],
               "est_answer_cost_usd": summary["est_answer_cost_usd"]}
    return {"arm_a": arm_row, "oracle": free["oracle_top30"], "top10": free["ceilings"]["top_10"],
            "best_cutoff": best_cutoff(free["score_cutoffs"]),
            "cutoff": free["score_cutoffs"]}


# =============================================================================
# Part 3: answer with the chosen setting, and the cache-only checks
# =============================================================================
CHOSEN_THRESHOLD = 0.7
CHOSEN_FLOOR = 1
CHOSEN_BUDGET = DEFAULT_BUDGET
CHOSEN_CANDIDATES = CANDIDATES
ANSWER_PROMPT_VERSION = "p2-relevance-v1"
RUN_NAME = "p07_f1_k30_b10000"


def chosen_passages(q: QuestionCandidates, probs: list[float], index: DocIndex, *,
                    threshold: float, floor: int, budget: int, n_candidates: int) -> tuple[list[str], list[str]]:
    """Passage texts and doc ids the filter sends for one question.

    The last passage is truncated when that is what the budget rule decided.
    """
    pool = q.candidates[:n_candidates]
    sent = select_by_probability(pool, probs[:n_candidates], threshold=threshold, budget=budget, floor=floor)
    texts: list[str] = []
    for cand, mode, n_tokens in sent:
        text = passage_text(cand, index)
        if mode.endswith("+cut"):
            text = truncate_to_tokens(text, n_tokens)
        texts.append(text)
    return texts, [cand.doc_id for cand, _, _ in sent]


def make_answer_fn(questions: list[QuestionCandidates], probs: dict[str, list[float]], index: DocIndex):
    """An `answer(question)` closure over the chosen filter setting."""
    # Built up front, in one thread: the split index is not safe to share across calls.
    prepared: dict[str, tuple[list[str], list[str]]] = {}
    for q in questions:
        prepared[q.question] = chosen_passages(
            q, probs[q.question_id], index, threshold=CHOSEN_THRESHOLD, floor=CHOSEN_FLOOR,
            budget=CHOSEN_BUDGET, n_candidates=CHOSEN_CANDIDATES)
    if len(prepared) != len(questions):
        raise RuntimeError("Two questions share the same text; answering keys on the text.")
    llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens)

    async def answer(question: str) -> dict[str, Any]:
        texts, doc_ids = prepared[question]
        res = await llm.ainvoke(messages(question, "\n\n".join(texts)), prompt_version=ANSWER_PROMPT_VERSION)
        return {"answer": res.text, "doc_ids": doc_ids}

    return answer


def answer_and_judge(questions: list[QuestionCandidates], probs: dict[str, list[float]]) -> Path:
    """Answer all 500 with luna and score them with Jev only. Returns the run directory."""
    out_dir = settings.paths.results_dir / "phase_2" / RUN_NAME
    fn = make_answer_fn(questions, probs, DocIndex())
    args = argparse.Namespace(limit=None, question_type=None, dry_run=False, evaluate=True,
                              concurrency=5, only_empty=False)
    # judge="jev": development runs are not scored by the cascade.
    code = run_phase(RUN_NAME, fn, args, out_dir=out_dir, judge="jev")
    if code not in (0, 1):
        raise RuntimeError(f"Answering exited {code}")
    return out_dir


def report_against_arm_a(run_dir: Path) -> dict[str, Any]:
    """Overall, per type, and flips versus arm A's Jev-judged answers."""
    arm_dir = settings.paths.results_dir / "phase_2" / "n5_w1_t2000_b8000"
    arm = _load_eval(arm_dir / "jev" / "answers_eval.jsonl")
    here = _load_eval(run_dir / "jev" / "answers_eval.jsonl")
    metrics = json.loads((run_dir / "jev" / "answers_metrics.json").read_text())
    arm_metrics = json.loads((arm_dir / "jev" / "answers_metrics.json").read_text())
    cost = json.loads((run_dir / "answer_cost.json").read_text())

    flips: dict[str, Counter] = defaultdict(Counter)
    for qid, row in here.items():
        before, after = bool(arm[qid]["correct"]), bool(row["correct"])
        kind = "wrong_to_right" if (not before and after) else "right_to_wrong" if (before and not after) else "same"
        flips[row["question_type"]][kind] += 1
        flips["OVERALL"][kind] += 1
    return {"metrics": metrics, "arm_metrics": arm_metrics, "flips": {k: dict(v) for k, v in flips.items()},
            "answer_cost_usd": cost.get("total_cost_usd"), "seconds": cost.get("seconds")}


def _load_eval(path: Path) -> dict[str, dict[str, Any]]:
    return {r["question_id"]: r for r in map(json.loads, path.open(encoding="utf-8"))}


def filter_call_cost(questions: list[QuestionCandidates], *, n_candidates: int) -> dict[str, Any]:
    """Dollars the v1 per-passage Jev calls cost for the first `n_candidates` passages.

    Read back from the cache, so this is the real billed input of those calls
    and makes no new requests. v2 batches are a different cache key.
    """
    jev = CachedJev()
    spec = relevance_questions()
    index = DocIndex()
    tokens = calls = missing = 0
    for q in questions:
        for cand in q.candidates[:n_candidates]:
            hit = jev.peek({"question": q.question, "passage": passage_text(cand, index)},
                           spec, prompt_version=RELEVANCE_PROMPT_VERSION_V1)
            if hit is None:
                missing += 1
                continue
            tokens += int(hit.get("input_tokens") or 0)
            calls += 1
    price = settings.model_prices.get(settings.jev_model) or settings.model_prices.get("jev-latest")
    usd = None if price is None else round(tokens * price.input / 1e6, 4)
    return {"calls": calls, "missing": missing, "input_tokens": tokens, "cost_usd": usd}


def extra_checks(questions: list[QuestionCandidates], probs: dict[str, list[float]]) -> dict[str, Any]:
    """The four cache-only checks. No model calls."""
    index = DocIndex()
    rejected: dict[str, list[float]] = defaultdict(list)
    gold_in_pool: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for q in questions:
        gold = set(q.gold)
        for cand, p in zip(q.candidates[:CHOSEN_CANDIDATES], probs[q.question_id][:CHOSEN_CANDIDATES]):
            if cand.doc_id not in gold:
                continue
            gold_in_pool[q.question_type] += 1
            if p < CHOSEN_THRESHOLD:
                rejected[q.question_type].append(p)
                if q.question_type in ("project_related", "completeness"):
                    examples.append({
                        "question_id": q.question_id, "question_type": q.question_type,
                        "question": q.question, "doc_id": cand.doc_id, "source_type": cand.source_type,
                        "probability": round(p, 4), "passage_tokens": cand.passage_tokens,
                        "passage": passage_text(cand, index),
                    })

    by_type = {}
    for qtype in sorted(set(gold_in_pool) | set(rejected)):
        got = rejected.get(qtype, [])
        pool_n = gold_in_pool[qtype]
        by_type[qtype] = {
            "rejected": len(got), "gold_in_top30": pool_n,
            "share_pct": round(100 * len(got) / pool_n, 1) if pool_n else None,
            "median_probability": round(statistics.median(got), 3) if got else None,
        }

    # Five lowest-probability rejections from each of the two types the cutoff hurt.
    shown = []
    for qtype in ("project_related", "completeness"):
        group = sorted((e for e in examples if e["question_type"] == qtype), key=lambda e: e["probability"])
        shown.extend(group[:5])
    (OUT_DIR / "rejected_gold_examples.json").write_text(json.dumps(shown, indent=2, ensure_ascii=False))

    # Miss rate: gold documents a question needed that the top 30 did not contain,
    # over every gold document of that source the questions ask for.
    needed: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    for q in questions:
        found = {c.doc_id for c in q.candidates[:CHOSEN_CANDIDATES]}
        for doc_id in dedupe(q.gold):
            source = source_of_gold(doc_id)
            needed[source] += 1
            missed[source] += doc_id not in found
    lengths = _gold_lengths_by_source()
    miss_rate = {
        source: {"missed": missed[source], "gold_asked_for": needed[source],
                 "miss_pct": round(100 * missed[source] / needed[source], 1),
                 "median_tokens": lengths.get(source)}
        for source in sorted(needed, key=lambda s: -missed[s])
    }

    empty_with_gold = 0
    questions_with_gold = 0
    for q in questions:
        if not q.gold:
            continue
        questions_with_gold += 1
        sent = select_by_probability(q.candidates[:CHOSEN_CANDIDATES], probs[q.question_id][:CHOSEN_CANDIDATES],
                                     threshold=CHOSEN_THRESHOLD, budget=CHOSEN_BUDGET, floor=0)
        empty_with_gold += not sent

    kept20 = {}
    for q in questions:
        sent = select_by_probability(q.candidates[:20], probs[q.question_id][:20],
                                     threshold=CHOSEN_THRESHOLD, budget=CHOSEN_BUDGET, floor=CHOSEN_FLOOR)
        kept20[q.question_id] = [c.doc_id for c, _, _ in sent]
    at20 = summarise_kept(questions, kept20)
    cost20 = filter_call_cost(questions, n_candidates=20)
    cost30 = filter_call_cost(questions, n_candidates=30)

    return {
        "rejected_gold_by_type": by_type,
        "rejected_gold_total": sum(v["rejected"] for v in by_type.values()),
        "gold_in_top30_total": sum(gold_in_pool.values()),
        "examples": shown,
        "miss_rate_by_source": miss_rate,
        "floor0_empty_with_gold": empty_with_gold,
        "questions_with_gold": questions_with_gold,
        "candidates_20": {"overall": at20["overall"], "filter_cost": cost20},
        "candidates_30_filter_cost": cost30,
    }


def _gold_lengths_by_source() -> dict[str, int]:
    """Median token length of the gold documents of each source."""
    from phase_1_baseline.ingest_naive import load_docs

    by_source: dict[str, list[int]] = defaultdict(list)
    for row in load_docs(None).itertuples():
        if row.is_gold:
            by_source[row.source_type].append(count_tokens(f"{row.title}\n\n{row.content}"))
    return {source: sorted(vals)[len(vals) // 2] for source, vals in by_source.items()}


def print_extra(checks: dict[str, Any]) -> None:
    """Print the four cache-only checks."""
    print("\n=== Gold in the top 30 that Jev scored below 0.7 ===")
    print(f"{checks['rejected_gold_total']} of {checks['gold_in_top30_total']} gold passages in the pool")
    print(f"{'type':28s}{'rejected':>10s}{'in pool':>10s}{'share':>8s}{'median p':>10s}")
    for qtype, row in checks["rejected_gold_by_type"].items():
        median = "-" if row["median_probability"] is None else f"{row['median_probability']:.3f}"
        print(f"{qtype:28s}{row['rejected']:10d}{row['gold_in_top30']:10d}{row['share_pct']:7.1f}%{median:>10s}")
    print("\nLowest-scoring rejected gold passages (full text in rejected_gold_examples.json):")
    for e in checks["examples"]:
        excerpt = e["passage"].replace("\n", " ")
        if len(excerpt) > 500:
            excerpt = excerpt[:500] + " ..."
        print(f"\n[{e['question_type']}] {e['question_id']} p={e['probability']} {e['source_type']} "
              f"({e['passage_tokens']} tok)\nQ: {e['question']}\n{excerpt}")

    print("\n=== Miss rate by source: gold the top 30 did not contain ===")
    print(f"{'source':16s}{'missed':>8s}{'asked':>8s}{'miss%':>8s}{'median tok':>12s}")
    for source, row in checks["miss_rate_by_source"].items():
        print(f"{source:16s}{row['missed']:8d}{row['gold_asked_for']:8d}{row['miss_pct']:7.1f}%"
              f"{row['median_tokens']:12d}")

    print(f"\n=== Floor 0: questions that have gold documents and get an empty prompt: "
          f"{checks['floor0_empty_with_gold']} of {checks['questions_with_gold']} ===")
    at20 = checks["candidates_20"]["overall"]
    c20, c30 = checks["candidates_20"]["filter_cost"], checks["candidates_30_filter_cost"]
    print("\n=== 20 candidates instead of 30 (same p>=0.7, floor 1, 10k budget) ===")
    print(f"recall@10 {at20['recall10']}  invalid {at20['invalid']}  docs avg {at20['avg_docs']}")
    print(f"filter cost, 20 candidates: ${c20['cost_usd']} ({c20['calls']} calls, {c20['missing']} uncached)")
    print(f"filter cost, 30 candidates: ${c30['cost_usd']} ({c30['calls']} calls)")


def print_comparison(report: dict[str, Any]) -> None:
    """Part 3 table: this run against arm A."""
    here, arm = report["metrics"], report["arm_metrics"]
    print("\n=== Answered: p>=0.7, floor 1, 10k, 30 candidates. Judge: Jev ===")
    print(f"{'':28s}{'overall':>10s}{'correct':>10s}{'complete':>10s}{'recall':>10s}{'invalid':>10s}{'words':>8s}")
    for label, block in (("arm A", arm), ("filter", here)):
        o = block["overall"]
        print(f"{label:28s}{o['overall_score']:10.2f}{o['correctness']:10.2f}{o['completeness']:10.2f}"
              f"{o['recall_at_k']:10.2f}{o['invalid_extra_docs']:10.2f}{o['avg_answer_words']:8.1f}")
    print(f"\nanswer cost ${report['answer_cost_usd']} in {report['seconds']}s")
    print("\nby question type (filter / arm A overall), and flips filter vs arm A:")
    print(f"{'type':28s}{'filter':>8s}{'arm A':>8s}{'wrong->right':>14s}{'right->wrong':>14s}")
    flips = report["flips"]
    for qtype in sorted(here["by_type"]):
        f, a = here["by_type"][qtype], arm["by_type"][qtype]
        flip = flips.get(qtype, {})
        print(f"{qtype:28s}{f['overall_score']:8.2f}{a['overall_score']:8.2f}"
              f"{flip.get('wrong_to_right', 0):14d}{flip.get('right_to_wrong', 0):14d}")
    overall = flips["OVERALL"]
    print(f"{'OVERALL':28s}{here['overall']['overall_score']:8.2f}{arm['overall']['overall_score']:8.2f}"
          f"{overall.get('wrong_to_right', 0):14d}{overall.get('right_to_wrong', 0):14d}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2.2: relevance filter over arm-A passages.")
    p.add_argument("--free", action="store_true", help="Part 1 only: ceilings, oracle, score cutoffs. $0.")
    p.add_argument("--dry-run", action="store_true", help="Price the Jev calls without making them.")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N questions (smoke test).")
    p.add_argument("--sweep", action="store_true", help="Threshold table from saved Jev scores; no calls.")
    p.add_argument("--candidates", type=int, default=CANDIDATES, help="Documents the filter may choose from.")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Max tokens of passages sent.")
    p.add_argument("--threshold", type=float, default=None, help="Single threshold (the sweep uses 0.3/0.5/0.7).")
    p.add_argument("--floor", type=int, default=None, choices=(0, 1), help="Minimum documents kept.")
    p.add_argument("--checks", action="store_true",
                   help="The four cache-only checks (rejected gold, miss rate, floor 0, 20 candidates). $0.")
    p.add_argument("--answer", action="store_true",
                   help="Part 3: answer all 500 at p>=0.7, floor 1, 10k, 30 candidates, and judge with Jev.")
    return p.parse_args(argv)


def _store_and_index():
    """The Qdrant store and the local re-split of every document. No answer model."""
    from phase_1_baseline.ingest_naive import COLLECTION
    from shared_utils.llm import CachedEmbeddings
    from shared_utils.vectorstore import dense_store, get_client

    client = get_client()
    if not client.collection_exists(COLLECTION):
        raise RuntimeError(f"Collection {COLLECTION} missing. Run phase_1_baseline.ingest_naive first.")
    return dense_store(client, COLLECTION, CachedEmbeddings()), DocIndex()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    try:
        if args.checks or args.answer:
            questions, probs = load_candidates(), load_scores()
            checks = extra_checks(questions, probs)
            (OUT_DIR / "extra_checks.json").write_text(json.dumps(
                {k: v for k, v in checks.items() if k != "examples"}, indent=2))
            print_extra(checks)
            if not args.answer:
                return 0
            run_dir = answer_and_judge(questions, probs)
            comparison = report_against_arm_a(run_dir)
            (run_dir / "vs_arm_a.json").write_text(json.dumps(comparison, indent=2))
            print_comparison(comparison)
            return 0

        if args.free or not CANDIDATES_PATH.is_file():
            store, index = _store_and_index()
            questions = build_all(store, index, load_qa())
            save_candidates(questions)
            logger.info("Wrote %s", CANDIDATES_PATH)
        else:
            questions = load_candidates()

        report = free_checks(questions)
        (OUT_DIR / "free_checks.json").write_text(json.dumps(report, indent=2))
        print_free(report)
        if args.free:
            return 0

        scored = questions if args.limit is None else questions[: args.limit]
        if not args.sweep:
            usage.reset()
            probs = asyncio.run(score_passages(scored, dry_run=args.dry_run))
            for line in usage.summary_lines():
                print(f"[cost] {line}")
            if args.dry_run:
                return 0
            if args.limit is not None:
                print("\n=== Smoke test: probability spread ===")
                print(json.dumps(probability_summary(scored, probs), indent=2))
                return 0
            save_scores(questions, probs, SCORES_PATH_V2)
            logger.info("Wrote %s. Left %s (the Phase 2.2 scores) unchanged.",
                        SCORES_PATH_V2.name, SCORES_PATH.name)
            sweep_path = OUT_DIR / "sweep_v2.json"
        else:
            probs = load_scores()
            sweep_path = OUT_DIR / "sweep.json"

        grid = sweep(questions, probs, budget=args.budget)
        sweep_path.write_text(json.dumps(grid, indent=2))
        refs = arm_a_and_references(report)
        print("\n=== References ===")
        print(f"{'arm A (N=5, no filter)':22s} {_fmt(refs['arm_a'])}  prompt {refs['arm_a']['prompt_tokens_avg']} tok")
        print(f"{'oracle (gold in top 30)':22s} {_fmt(refs['oracle'])}")
        # Keeping the top 10 beats every score cutoff on Recall@10, so it is the
        # free baseline the filter has to improve on.
        print(f"{'top 10 documents':22s} {_fmt(refs['top10'])}")
        print(f"{refs['best_cutoff'] + ' (best cutoff)':22s} {_fmt(refs['cutoff'][refs['best_cutoff']])}")
        print_sweep_grid(grid)
        from shared_utils.evaluation import append_run_log
        append_run_log({"kind": "relevance_filter", "questions": len(questions), "budget": args.budget,
                        "sweep": {name: stats["overall"] for name, stats in grid.items()},
                        "output": str(sweep_path)})
        return 0
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


def print_sweep_grid(grid: dict[str, Any]) -> None:
    print("\n=== Jev filter: threshold x floor ===")
    for name, stats in grid.items():
        print(f"{name:16s} {_fmt(stats)}  prompt {stats['prompt_tokens_avg']} tok  "
              f"est ${stats['est_answer_cost_usd']}  info_not_found empty {stats['info_not_found_zero_docs_pct']}%")
    print("\nset recall / avg docs, by question type:")
    qtypes = list(next(iter(grid.values()))["by_type"])
    print(f"{'type':28s}" + "".join(f"{n:>16s}" for n in grid))
    for qtype in qtypes:
        cells = []
        for stats in grid.values():
            b = stats["by_type"][qtype]
            cells.append(f"{b['set_recall']}/{b['avg_docs']}")
        print(f"{qtype:28s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    sys.exit(main())
