#!/usr/bin/env python3
"""Step 1 / Phase 0: curate "mini-redwood", a ~4.7k-document slice of EnterpriseRAG-Bench.

Why a subset?
    The full corpus is ~512k documents (3.2 GB). Embedding and indexing it for
    every experiment would be slow and would cost real money on every change.
    mini-redwood keeps *every* document any benchmark question needs (the
    "gold" set) plus a realistic dose of distracting "noise", so retrieval is
    still hard, but a full ingest + evaluation cycle takes minutes and cents.

What goes in:
    * Gold  - every document listed in any question's `expected_doc_ids`
              (722 docs across all 9 source types).
    * Noise - 4,000 random non-gold docs from slack, gmail, confluence and
              linear: the sources where near-duplicates, jargon, misfiled docs
              and conflicting facts live, i.e. what makes naive RAG fail.

Caveat worth knowing when reading scores:
    "High Level" questions have no gold documents (they need broad corpus
    knowledge), so on mini-redwood their supporting evidence is only present
    if it happens to be in the noise sample. Expect them to under-perform
    relative to the full corpus.

Inputs:
    all_documents/                   unzipped `all_documents.zip` from the Onyx release
        <source_type>/<nested dirs>/dsid_<32 hex>__<slug>.txt
        (first line = title -- or the channel name for Slack; rest = content)
    all_documents/questions.jsonl    the 500 benchmark questions (default source)
    Hugging Face `onyx-dot-app/EnterpriseRAG-Bench` "questions" config (--questions-source hf)

Outputs:
    data/raw_onyx_subset/mini_redwood_docs.jsonl
        {doc_id, source_type, title, content, path, is_gold, selection}
        `selection` is "gold" or "noise" here; scripts/add_distractors.py later
        appends "distractor" and "high_level_evidence" rows. Re-running this
        script resets the file, so run add_distractors.py again afterwards.
        `path` (relative to all_documents/) is the unique row key: a few doc_ids
        are shared by two different files in the original corpus.
    data/raw_onyx_subset/mini_redwood_qa.jsonl
        the 500 questions, unchanged

Cost: $0. No paid API is called; the only network use is the optional
Hugging Face download of the (400 KB) questions file.

Usage:
    python scripts/curate_mini_redwood.py
    python scripts/curate_mini_redwood.py --questions-source hf
    python scripts/curate_mini_redwood.py --allocation proportional --noise-count 4000 --seed 7
    python scripts/curate_mini_redwood.py --dry-run        # show the plan, write nothing
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/curate_mini_redwood.py` from the repo root: put the repo
# root on sys.path so `shared_utils` is importable without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_utils.config import settings  # noqa: E402

HF_DATASET = "onyx-dot-app/EnterpriseRAG-Bench"
HF_SPLIT = "test"

# Filenames look like `dsid_c5301621d7e04b7f870d3436148d784e__some-slug.txt`.
# The 32-hex part after `dsid_` is exactly the `doc_id` used in `expected_doc_ids`.
DOC_ID_RE = re.compile(r"^(dsid_[0-9a-f]{32})__")

logger = logging.getLogger("curate_mini_redwood")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line flags. Defaults come from shared_utils.config."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", type=Path, default=settings.paths.corpus_dir,
                   help="Unzipped all_documents directory.")
    p.add_argument("--questions-source", choices=("local", "hf"), default="local",
                   help="'local' reads all_documents/questions.jsonl (free, offline); "
                        "'hf' downloads the Hugging Face 'questions' config. Either falls back to the other.")
    p.add_argument("--noise-count", type=int, default=settings.noise_count,
                   help="Number of noise documents to sample.")
    p.add_argument("--noise-sources", nargs="+", default=settings.noise_sources,
                   help="Source types to draw noise from.")
    p.add_argument("--allocation", choices=("equal", "proportional"), default="equal",
                   help="'equal' = same number per noise source (1,000 each by default); "
                        "'proportional' = follow each source's share of the corpus (Slack-heavy).")
    p.add_argument("--seed", type=int, default=settings.seed, help="Random seed (sampling + shuffle).")
    p.add_argument("--output-dir", type=Path, default=settings.paths.subset_dir,
                   help="Where the two JSONL files are written.")
    p.add_argument("--dry-run", action="store_true",
                   help="Index the corpus and print what would be selected, but write nothing.")
    return p.parse_args()


# =============================================================================
# Questions -> gold doc IDs
# =============================================================================
def load_questions_local(path: Path) -> pd.DataFrame:
    """Read the release's questions.jsonl (one JSON object per line)."""
    if not path.is_file():
        raise FileNotFoundError(f"Local questions file not found: {path}")
    return pd.read_json(path, lines=True)


def load_questions_hf() -> pd.DataFrame:
    """Download the Hugging Face 'questions' config (~400 KB) as a DataFrame.

    `datasets` is imported lazily so the default (local) path does not pay its
    import time.
    """
    from datasets import load_dataset

    return load_dataset(HF_DATASET, "questions", split=HF_SPLIT).to_pandas()


def load_questions(source: str) -> pd.DataFrame:
    """Load questions from the preferred source, falling back to the other one.

    Both sources contain the same 500 questions; the fallback just means a
    missing file or a network hiccup does not stop the run.
    """
    loaders = {"local": lambda: load_questions_local(settings.paths.local_questions), "hf": load_questions_hf}
    order = [source, "hf" if source == "local" else "local"]
    last_exc: Exception | None = None
    for name in order:
        try:
            df = loaders[name]()
            logger.info("Loaded %d questions from %s.", len(df), name)
            break
        except Exception as exc:  # noqa: BLE001 - we log and try the fallback
            logger.warning("Could not load questions from %s: %s", name, exc)
            last_exc = exc
    else:
        raise RuntimeError("Could not load questions from any source.") from last_exc

    missing = {"question_id", "question", "question_type", "expected_doc_ids"} - set(df.columns)
    if missing:
        raise ValueError(f"Questions are missing expected columns: {sorted(missing)}")
    return df


def extract_gold_ids(questions: pd.DataFrame) -> set[str]:
    """Union of every question's `expected_doc_ids`.

    "Info Not Found" and "High Level" questions have an empty list; they simply
    contribute nothing. Values may be Python lists (local JSON) or numpy arrays
    (Hugging Face), so we iterate generically.
    """
    gold: set[str] = set()
    for ids in questions["expected_doc_ids"]:
        if ids is not None:
            gold.update(str(i) for i in ids if i)
    return gold


# =============================================================================
# Corpus index (paths only -- no file contents are read here)
# =============================================================================
def build_index(source_dir: Path) -> pd.DataFrame:
    """Walk the corpus and return one row per document: doc_id, source_type, path.

    Why only paths? Reading 3.2 GB of text to then keep ~1% of it would be
    wasteful. File names already carry the doc_id and the top-level folder is
    the source type, so the full index is built from directory metadata alone
    (`os.scandir` avoids an extra stat() per entry). We then read only the
    ~4.7k selected files.

    Args:
        source_dir: The unzipped `all_documents` directory.

    Returns:
        DataFrame with columns doc_id, source_type, path (relative to source_dir).

    Raises:
        FileNotFoundError: if the directory is missing or contains no documents.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {source_dir}. Download all_documents.zip from "
            "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/latest and unzip it there."
        )

    rows: list[tuple[str, str, str]] = []
    skipped = 0
    # Top-level folders are the source types (slack, gmail, ...). Files at the
    # top level (e.g. questions.jsonl) are not documents and are ignored.
    for source_entry in sorted(os.scandir(source_dir), key=lambda e: e.name):
        if not source_entry.is_dir():
            continue
        source_type = source_entry.name
        stack = [source_entry.path]
        while stack:  # iterative DFS: no recursion-depth concerns on deep trees
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.name.endswith(".txt"):
                        m = DOC_ID_RE.match(entry.name)
                        if m:
                            rows.append((m.group(1), source_type, os.path.relpath(entry.path, source_dir)))
                        else:
                            skipped += 1

    if not rows:
        raise FileNotFoundError(f"No dsid_*.txt documents found under {source_dir}.")
    if skipped:
        logger.warning("Skipped %d .txt files whose names do not match dsid_<32 hex>__*.txt.", skipped)

    index = pd.DataFrame(rows, columns=["doc_id", "source_type", "path"])
    # The corpus contains a handful of doc_ids shared by two *different* files
    # (e.g. a Jira ticket and its misfiled variant, or a HubSpot record and a
    # Confluence brief). These are the benchmark's deliberate near-duplicate /
    # misfiling noise, not copies -- so we keep every file. Downstream code must
    # therefore key chunks by `path`, not by `doc_id`.
    shared = index[index["doc_id"].duplicated(keep=False)]
    if not shared.empty:
        logger.info(
            "%d doc_ids are shared by %d distinct files; keeping all of them (key by `path`).",
            shared["doc_id"].nunique(), len(shared),
        )

    logger.info(
        "Indexed %d documents. Per source:\n%s",
        len(index),
        index["source_type"].value_counts().to_string(),
    )
    return index


# =============================================================================
# Noise sampling
# =============================================================================
def allocate_quotas(available: pd.Series, total: int, allocation: str) -> dict[str, int]:
    """Split a `total` noise budget across sources, capped by availability.

    Args:
        available:  Count of eligible (non-gold) docs per source type.
        total:      Number of noise docs wanted.
        allocation: "equal" (same share each) or "proportional" (by corpus share).

    Returns:
        {source_type: number_to_sample}. Sums to `total` unless the corpus
        simply does not have that many eligible docs.

    Rounding remainders go to the sources with the largest fractional share,
    and any budget a small source cannot fill is redistributed to the others,
    so we always hit `total` when possible.
    """
    available = available[available > 0]
    if available.empty or total <= 0:
        return {}

    if allocation == "proportional":
        weights = available / available.sum()
    else:
        weights = pd.Series(1.0 / len(available), index=available.index)

    exact = weights * total
    quotas = exact.astype(int)
    remainder = total - int(quotas.sum())
    for source in (exact - quotas).sort_values(ascending=False).index[:remainder]:
        quotas[source] += 1
    quotas = quotas.clip(upper=available)

    shortfall = total - int(quotas.sum())
    while shortfall > 0:
        room = (available - quotas)[lambda s: s > 0]
        if room.empty:
            break
        share = max(1, shortfall // len(room))
        for source in room.index:
            add = int(min(share, room[source], shortfall))
            quotas[source] += add
            shortfall -= add
            if shortfall == 0:
                break

    return {source: int(q) for source, q in quotas.items()}


def sample_noise(index: pd.DataFrame, gold_ids: set[str], sources: list[str], count: int,
                 allocation: str, seed: int) -> pd.DataFrame:
    """Stratified random sample of non-gold docs from the given source types.

    Stratifying (rather than sampling uniformly) guarantees each noisy source is
    represented; a uniform sample would be ~64% Slack and give Confluence only a
    few dozen docs.
    """
    candidates = index[index["source_type"].isin(sources) & ~index["doc_id"].isin(gold_ids)]

    absent = set(sources) - set(candidates["source_type"])
    if absent:
        logger.warning("No noise candidates for source types: %s", sorted(absent))

    available = candidates["source_type"].value_counts()
    if available.sum() < count:
        logger.warning("Requested %d noise docs but only %d are eligible.", count, int(available.sum()))

    quotas = allocate_quotas(available, count, allocation)
    logger.info("Noise quotas per source (%s): %s", allocation, quotas)

    parts = [candidates[candidates["source_type"] == s].sample(n=n, random_state=seed)
             for s, n in quotas.items() if n > 0]
    return pd.concat(parts) if parts else candidates.iloc[0:0]


# =============================================================================
# Reading selected documents
# =============================================================================
def read_document(source_dir: Path, rel_path: str) -> tuple[str, str]:
    """Read one exported document and split it into (title, content).

    The Onyx export puts the title on the first line (for Slack it is the
    channel name), usually followed by a blank line, then the body. We keep the
    title separately because it is high-signal metadata for retrieval later.

    `errors="replace"` means a stray invalid byte cannot crash the whole run;
    it becomes U+FFFD in that one document instead.
    """
    text = (source_dir / rel_path).read_text(encoding="utf-8", errors="replace")
    title, _, body = text.partition("\n")
    return title.strip(), body.strip()


def load_documents(source_dir: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Attach title/content to the selected index rows. Unreadable files are dropped (and logged)."""
    titles, contents, keep = [], [], []
    for i, rel in zip(rows.index, rows["path"]):
        try:
            title, content = read_document(source_dir, rel)
        except OSError as exc:
            logger.warning("Could not read %s: %s", rel, exc)
            continue
        keep.append(i)
        titles.append(title)
        contents.append(content)
    out = rows.loc[keep].copy()
    out["title"] = titles
    out["content"] = contents
    return out


# =============================================================================
# Output
# =============================================================================
def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Write `df` as JSON Lines atomically.

    We write to a temp file and `os.replace` it into place, so an interrupted
    run never leaves a half-written file that later phases would silently ingest.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_json(tmp, orient="records", lines=True, force_ascii=False)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("Wrote %d records to %s", len(df), path)


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    """Run the curation pipeline. Returns a process exit code."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.noise_count < 0:
        logger.error("--noise-count must be >= 0")
        return 2

    try:
        # 1. Questions -> gold IDs.
        questions = load_questions(args.questions_source)
        gold_ids = extract_gold_ids(questions)
        logger.info("%d unique gold doc IDs across %d questions.", len(gold_ids), len(questions))

        # 2. Index the full local corpus (paths only).
        index = build_index(args.source_dir)

        # 3. Gold documents.
        gold_rows = index[index["doc_id"].isin(gold_ids)]
        missing_gold = gold_ids - set(gold_rows["doc_id"])
        if missing_gold:
            logger.warning("%d gold IDs not found in the corpus, e.g. %s",
                           len(missing_gold), sorted(missing_gold)[:5])
        logger.info("Matched %d / %d gold doc IDs (%d files).",
                    gold_rows["doc_id"].nunique(), len(gold_ids), len(gold_rows))

        # 4. Noise documents (never gold).
        noise_rows = sample_noise(index, gold_ids, args.noise_sources, args.noise_count,
                                  args.allocation, args.seed)
        logger.info("Sampled %d noise documents.", len(noise_rows))

        selected = pd.concat([gold_rows.assign(is_gold=True, selection="gold"),
                              noise_rows.assign(is_gold=False, selection="noise")])
        mix = selected.groupby(["source_type", "is_gold"]).size().unstack(fill_value=0)
        logger.info("Selection (%d docs) by source and gold flag:\n%s", len(selected), mix.to_string())

        if args.dry_run:
            logger.info("--dry-run: nothing written.")
            return 0

        # 5. Read only the selected files, shuffle, save.
        docs = load_documents(args.source_dir, selected)
        # Shuffle so gold docs are not clustered at the top of the file; any
        # accidental "first N docs" shortcut in later code would otherwise be
        # biased toward gold and inflate scores.
        docs = docs.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        docs = docs[["doc_id", "source_type", "title", "content", "path", "is_gold", "selection"]]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(docs, args.output_dir / settings.paths.mini_docs.name)

        # 6. Questions, unchanged, for evaluation.
        write_jsonl(questions, args.output_dir / settings.paths.mini_qa.name)

        empty = int((docs["content"].str.len() == 0).sum())
        if empty:
            logger.warning("%d selected documents have empty content.", empty)

    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except OSError as exc:
        logger.error("File I/O error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted.")
        return 130

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
