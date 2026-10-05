"""Free check: would BM25, or fusing it with the current vector search, find more gold?

No model is called. The vector side is the candidate list already saved by the
relevance filter (top 50 documents, from the top 200 cosine chunks). The BM25
side is a local Okapi index over the same chunks Phase 1 embedded, so the two
lists are rankings of the same passages. Documents are scored by their best
chunk, which is how the cosine search already ranks them.

Reciprocal rank fusion (k = 60, the constant Qdrant uses) combines each
method's top 50 documents. A document missing from one list simply gets no
term from that list.

Usage:
    python -m phase_2_data_mastery.hybrid_check
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.ingest_naive import chunk_documents, load_docs  # noqa: E402
from phase_2_data_mastery.relevance_filter import load_candidates, source_of_gold  # noqa: E402
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import dedupe  # noqa: E402

# Depth of each list. 30 is the filter's pool; 50 is as deep as the saved
# vector search goes, so fusion cannot look further than either method did.
TOPS = (30, 50)
# Cormack, Clarke, Buettcher 2009. Also Qdrant's default RRF constant.
RRF_K = 60
# Okapi BM25. k1 saturates term frequency; b scales by document length.
BM25_K1 = 1.5
BM25_B = 0.75
# Keep hyphenated ids (ADR-007, PROJ-123) as one token. That is the lexical
# signal dense search is weakest on.
TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
OUT_DIR = settings.paths.results_dir / "phase_2" / "hybrid_check"


def tokenize(text: str) -> list[str]:
    """Lowercased tokens, with hyphenated identifiers kept intact."""
    return TOKEN.findall(text.lower())


class BM25Index:
    """In-memory Okapi BM25 over one token list per passage.

    IDF is the always-positive form used by Lucene and by `rank_bm25`:
    log((N - df + 0.5) / (df + 0.5) + 1). A term in every passage scores 0.
    """

    def __init__(self, tokenized: list[list[str]]):
        self.k1 = BM25_K1
        self.b = BM25_B
        self.n = len(tokenized)
        self.doc_len = [len(tokens) for tokens in tokenized]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        df: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(tokenized):
            counts = Counter(tokens)
            df.update(counts.keys())
            for term, tf in counts.items():
                postings[term].append((i, tf))
        self.postings = postings
        self.idf = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def rank_documents(self, query: str, meta: list[dict], *, limit: int) -> list[dict]:
        """Best `limit` documents for `query`, highest BM25 first.

        A document's score is its best passage. Two files that share a doc_id
        collapse to the higher score, matching the vector candidate list.
        """
        scores: dict[int, float] = defaultdict(float)
        seen: set[str] = set()
        for term in tokenize(query):
            if term in seen or term not in self.idf:
                continue
            seen.add(term)
            idf = self.idf[term]
            for index, tf in self.postings[term]:
                length = self.doc_len[index]
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                scores[index] += idf * tf * (self.k1 + 1) / denom

        best: dict[str, dict] = {}
        for index, score in scores.items():
            row = meta[index]
            current = best.get(row["doc_id"])
            if current is None or score > current["score"]:
                best[row["doc_id"]] = {**row, "score": score}
        ranked = sorted(best.values(), key=lambda row: -row["score"])
        return ranked[:limit]

    def ranked_documents(self, query: str, meta: list[dict], *, limit: int) -> list[dict]:
        """Like `rank_documents`, and also the chunks that matched, best first.

        Chunks stay with the file they came from. When two files share a
        doc_id, the higher-scoring file is kept, same as the vector search.
        `chunks` is what the small-to-big window is built around.
        """
        scores: dict[int, float] = defaultdict(float)
        seen: set[str] = set()
        for term in tokenize(query):
            if term in seen or term not in self.idf:
                continue
            seen.add(term)
            idf = self.idf[term]
            for index, tf in self.postings[term]:
                length = self.doc_len[index]
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                scores[index] += idf * tf * (self.k1 + 1) / denom

        by_path: dict[str, dict] = {}
        for index, score in scores.items():
            row = meta[index]
            entry = by_path.get(row["path"])
            if entry is None:
                entry = {**row, "score": score, "chunks": []}
                by_path[row["path"]] = entry
            entry["score"] = max(entry["score"], score)
            entry["chunks"].append((score, row["chunk_index"]))

        by_doc: dict[str, dict] = {}
        for entry in by_path.values():
            current = by_doc.get(entry["doc_id"])
            if current is None or entry["score"] > current["score"]:
                by_doc[entry["doc_id"]] = entry
        ranked = sorted(by_doc.values(), key=lambda row: -row["score"])
        for entry in ranked[:limit]:
            entry["chunks"] = [i for _, i in sorted(entry["chunks"], key=lambda pair: -pair[0])]
        return ranked[:limit]


def fuse(vector_ids: list[str], bm25_ids: list[str], *, limit: int) -> list[str]:
    """Reciprocal rank fusion. Ranks are 1-based. Missing lists contribute nothing."""
    scores: dict[str, float] = defaultdict(float)
    for ids in (vector_ids, bm25_ids):
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] += 1.0 / (RRF_K + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: -item[1])][:limit]


def gold_found(questions, lists: dict[str, list[str]], k: int) -> dict:
    """How many gold documents each question asked for sit in the top `k`.

    Counted once per question (repeated gold ids count once), which is the
    same denominator as the 142 vector misses.
    """
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_source: dict[str, Counter] = defaultdict(Counter)
    found = asked = 0
    per_question: list[float] = []
    for q in questions:
        gold = dedupe(q.gold)
        if not gold:
            continue
        hit = set(lists[q.question_id][:k])
        n_hit = sum(doc_id in hit for doc_id in gold)
        found += n_hit
        asked += len(gold)
        per_question.append(n_hit / len(gold))
        by_type[q.question_type]["found"] += n_hit
        by_type[q.question_type]["asked"] += len(gold)
        for doc_id in gold:
            source = source_of_gold(doc_id)
            by_source[source]["asked"] += 1
            by_source[source]["found"] += doc_id in hit
    return {
        "found": found, "asked": asked,
        "pct": round(100 * found / asked, 2) if asked else None,
        "per_question_pct": round(100 * sum(per_question) / len(per_question), 2) if per_question else None,
        "by_type": _share(by_type),
        "by_source": _share(by_source),
    }


def _share(counts: dict[str, Counter]) -> dict[str, dict]:
    out = {}
    for key in sorted(counts, key=lambda name: -counts[name]["asked"]):
        found, asked = counts[key]["found"], counts[key]["asked"]
        out[key] = {"found": found, "asked": asked, "pct": round(100 * found / asked, 1) if asked else None}
    return out


def rescue_and_loss(questions, vector: dict[str, list[str]], bm25: dict[str, list[str]],
                    fused: dict[str, list[str]], *, k: int) -> dict:
    """Of the gold vector missed in its top `k`: who else finds it, and what fusion drops."""
    missed = bm25_finds = fused_finds = fused_loses = vector_had = 0
    only_bm25: list[dict] = []
    only_vector: list[dict] = []
    for q in questions:
        vset = set(vector[q.question_id][:k])
        bset = set(bm25[q.question_id][:k])
        fset = set(fused[q.question_id][:k])
        for doc_id in dedupe(q.gold):
            in_v, in_b = doc_id in vset, doc_id in bset
            if not in_v:
                missed += 1
                bm25_finds += in_b
                fused_finds += doc_id in fset
            else:
                vector_had += 1
                fused_loses += doc_id not in fset
            if in_b and not in_v:
                only_bm25.append(_example(q, doc_id, bm25[q.question_id], "bm25"))
            elif in_v and not in_b:
                only_vector.append(_example(q, doc_id, vector[q.question_id], "vector"))
    return {
        "vector_missed": missed,
        "bm25_finds": bm25_finds,
        "fused_finds": fused_finds,
        "vector_had": vector_had,
        "fused_loses": fused_loses,
        "only_bm25": _pick(only_bm25),
        "only_vector": _pick(only_vector),
    }


def _example(q, doc_id: str, ranked: list[str], method: str) -> dict:
    return {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "question": q.question,
        "doc_id": doc_id,
        "source_type": source_of_gold(doc_id),
        "rank": ranked.index(doc_id) + 1,
        "method": method,
    }


def _pick(rows: list[dict], n: int = 5) -> list[dict]:
    """Five finds from five different questions, strongest rank first."""
    chosen = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: item["rank"]):
        if row["question_id"] in seen:
            continue
        seen.add(row["question_id"])
        chosen.append(row)
        if len(chosen) == n:
            break
    return chosen


def attach_documents(examples: list[dict]) -> None:
    """Add the gold document's title and opening, looked up from mini-redwood."""
    wanted = {row["doc_id"] for row in examples}
    found: dict[str, dict] = {}
    for row in load_docs(None).itertuples():
        if row.doc_id not in wanted:
            continue
        if row.doc_id in found and not row.is_gold:
            continue
        opening = f"{row.title}\n\n{row.content}".replace("\n", " ")
        found[row.doc_id] = {"title": row.title, "path": row.path, "opening": opening[:500]}
    for row in examples:
        row.update(found.get(row["doc_id"], {}))


def build_bm25():
    """Index every Phase 1 chunk. Returns the index and per-chunk metadata."""
    docs = load_docs(None)
    chunks, _ = chunk_documents(docs)
    meta = [
        {"doc_id": c.metadata["doc_id"], "path": c.metadata["path"],
         "source_type": c.metadata["source_type"], "title": c.metadata["title"],
         "chunk_index": c.metadata["chunk_index"]}
        for c in chunks
    ]
    index = BM25Index([tokenize(c.page_content) for c in chunks])
    return index, meta, len(chunks)


def main() -> int:
    setup_logging()
    questions = load_candidates()
    index, meta, n_chunks = build_bm25()

    vector = {q.question_id: [c.doc_id for c in q.candidates] for q in questions}
    bm25 = {
        q.question_id: [row["doc_id"] for row in index.rank_documents(q.question, meta, limit=max(TOPS))]
        for q in questions
    }
    fused = {
        qid: fuse(vector[qid][: max(TOPS)], bm25[qid][: max(TOPS)], limit=max(TOPS))
        for qid in vector
    }
    methods = {"vector": vector, "bm25": bm25, "fused": fused}

    report = {
        "passages": n_chunks,
        "bm25": {"k1": BM25_K1, "b": BM25_B},
        "rrf_k": RRF_K,
        "fusion_depth": max(TOPS),
        "gold_found": {str(k): {name: gold_found(questions, lists, k) for name, lists in methods.items()}
                       for k in TOPS},
        "top30": rescue_and_loss(questions, vector, bm25, fused, k=30),
    }
    attach_documents(report["top30"]["only_bm25"])
    attach_documents(report["top30"]["only_vector"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _print(report)
    return 0


def _print(report: dict) -> None:
    print(f"\nBM25 over {report['passages']} passages (same chunks as the vector index). "
          f"RRF k={report['rrf_k']} of each method's top {report['fusion_depth']}.\n")
    print(f"{'':22s}{'vector top30':>14s}{'BM25 top30':>14s}{'fused top30':>14s}"
          f"{'vector top50':>14s}{'BM25 top50':>14s}{'fused top50':>14s}")
    g30, g50 = report["gold_found"]["30"], report["gold_found"]["50"]
    v30, b30, f30 = g30["vector"], g30["bm25"], g30["fused"]
    v50, b50, f50 = g50["vector"], g50["bm25"], g50["fused"]
    print(f"{'gold found':22s}{_cell(v30)}{_cell(b30)}{_cell(f30)}{_cell(v50)}{_cell(b50)}{_cell(f50)}")
    print(f"{'per question':22s}"
          f"{v30['per_question_pct']:13.1f}%{b30['per_question_pct']:13.1f}%{f30['per_question_pct']:13.1f}%"
          f"{v50['per_question_pct']:13.1f}%{b50['per_question_pct']:13.1f}%{f50['per_question_pct']:13.1f}%")

    print("\nBy question type (gold documents found / asked for):")
    print(f"{'type':28s}{'v30':>10s}{'b30':>10s}{'f30':>10s}{'v50':>10s}{'b50':>10s}{'f50':>10s}")
    for qtype in v30["by_type"]:
        print(f"{qtype:28s}{_pct(v30, qtype)}{_pct(b30, qtype)}{_pct(f30, qtype)}"
              f"{_pct(v50, qtype)}{_pct(b50, qtype)}{_pct(f50, qtype)}")

    print("\nBy source type:")
    print(f"{'source':16s}{'v30':>10s}{'b30':>10s}{'f30':>10s}{'v50':>10s}{'b50':>10s}{'f50':>10s}")
    for source in v30["by_source"]:
        print(f"{source:16s}{_pct(v30, source, 'by_source')}{_pct(b30, source, 'by_source')}"
              f"{_pct(f30, source, 'by_source')}{_pct(v50, source, 'by_source')}"
              f"{_pct(b50, source, 'by_source')}{_pct(f50, source, 'by_source')}")

    top = report["top30"]
    print(f"\nOf the {top['vector_missed']} gold documents vector missed in the top 30:")
    print(f"  BM25 top 30 finds {top['bm25_finds']}")
    print(f"  fused top 30 finds {top['fused_finds']}")
    print(f"Of the {top['vector_had']} gold documents vector had in the top 30, "
          f"fusion drops {top['fused_loses']}.")

    for label, key in (("only BM25 finds", "only_bm25"), ("only vector finds", "only_vector")):
        print(f"\n=== 5 gold documents {label} (top 30) ===")
        for row in top[key]:
            print(f"\n[{row['question_type']}] {row['question_id']}  {row['source_type']}  "
                  f"rank {row['rank']}  {row['doc_id']}")
            print(f"Q: {row['question']}")
            print(row.get("opening", "")[:400])


def _cell(block: dict) -> str:
    return f"{block['found']}/{block['asked']} ({block['pct']:.1f}%)".rjust(14)


def _pct(block: dict, key: str, field: str = "by_type") -> str:
    row = block[field].get(key)
    if not row or row["pct"] is None:
        return f"{'—':>10s}"
    return f"{row['pct']:9.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
