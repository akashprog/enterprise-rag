"""Measure mini-redwood document lengths, to choose the small-to-big expansion rule from data.

Small-to-big (query_small_to_big.py) sends a whole document when it is "short"
and a window of chunks around the matches otherwise. Where "short" ends depends
on how long documents actually are, per source type: a Slack thread and a
Confluence spec are very different sizes. This script answers that question.

What it reports, overall, per source type, and for gold documents only (the ones
the answers come from, so the ones whose size matters most):
    * token count: median, p90, max (tiktoken, the answer model's tokenizer)
    * chunks per document under Phase 1's splitter (1,000 chars / 200 overlap)
    * share of documents at or under candidate whole-document thresholds

Tokens are counted on the same text Phase 1 chunks ("title\\n\\nbody"), so the
numbers translate directly into prompt tokens.

Cost: $0. Local tokenizer only, no API calls, no Qdrant.

Usage:
    python -m phase_2_data_mastery.measure_doc_lengths
    python -m phase_2_data_mastery.measure_doc_lengths --thresholds 500 1000 2000

Reads:  data/raw_onyx_subset/mini_redwood_docs.jsonl
Writes: results/analysis/doc_lengths.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.ingest_naive import CHUNK_OVERLAP, CHUNK_SIZE, load_docs  # noqa: E402
from shared_utils.config import settings  # noqa: E402
from shared_utils.llm import count_tokens  # noqa: E402

DEFAULT_THRESHOLDS = [500, 1000, 1500, 2000, 3000, 4000]


def measure(docs: pd.DataFrame) -> pd.DataFrame:
    """Add `tokens` and `chunks` columns (Phase 1 text and splitter) to `docs`."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = [f"{t}\n\n{c}" for t, c in zip(docs["title"], docs["content"])]
    return docs.assign(tokens=[count_tokens(t) for t in texts],
                       chunks=[len(splitter.split_text(t)) for t in texts])


def summarise(g: pd.DataFrame, thresholds: list[int]) -> dict[str, float]:
    """Length statistics for one group of documents."""
    tok = g["tokens"]
    row: dict[str, float] = {
        "docs": len(g),
        "median": int(tok.median()),
        "p90": int(tok.quantile(0.9)),
        "max": int(tok.max()),
        "chunks_median": int(g["chunks"].median()),
        "chunks_p90": int(g["chunks"].quantile(0.9)),
    }
    for t in thresholds:
        row[f"<= {t}"] = round(100 * float((tok <= t).mean()), 1)
    return row


def table(docs: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """One row for all documents, then one per source type (largest first)."""
    rows = {"ALL": summarise(docs, thresholds)}
    for src, g in sorted(docs.groupby("source_type"), key=lambda kv: -len(kv[1])):
        rows[src] = summarise(g, thresholds)
    return pd.DataFrame(rows).T


def main() -> int:
    p = argparse.ArgumentParser(description="Token lengths of mini-redwood documents (free).")
    p.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS,
                   help="Candidate whole-document thresholds (tokens) to report coverage for.")
    p.add_argument("--output", type=Path, default=settings.paths.results_dir / "analysis" / "doc_lengths.json")
    args = p.parse_args()

    docs = measure(load_docs(None))
    all_docs = table(docs, args.thresholds)
    gold = table(docs[docs["is_gold"]], args.thresholds)

    pd.set_option("display.width", 200)
    print("=== All documents (tokens; '<= N' = % of docs at or under N tokens) ===")
    print(all_docs.to_string())
    print("\n=== Gold documents only ===")
    print(gold.to_string())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"all": all_docs.to_dict("index"), "gold": gold.to_dict("index")}, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
