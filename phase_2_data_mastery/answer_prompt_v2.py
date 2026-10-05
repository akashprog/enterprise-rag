"""Step 2.4: a new answer prompt on DEV and the info_not_found guard.

Retrieval, the relevance filter, the passages, and the judge are Phase 2.3's.
Only the wording around those passages changes. HOLDOUT is not answered.

Two arms, both judged with Jev only:

    ARM 1  prompt v2, passages joined with a blank line, as in Phase 2.3
    ARM 2  the same prompt, plus a title and source label on each passage

Usage:
    python -m phase_2_data_mastery.answer_prompt_v2 --dry-run
    python -m phase_2_data_mastery.answer_prompt_v2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_2_data_mastery.hybrid_candidates import (  # noqa: E402
    FOCUS,
    GUARD,
    PASS_AT,
    RUN_NAME,
    load_fused,
    select_sent,
)
from phase_2_data_mastery.query_small_to_big import DocIndex  # noqa: E402
from phase_2_data_mastery.relevance_filter import passage_text  # noqa: E402
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import QuestionScore, aggregate, dedupe, load_answers, load_qa  # noqa: E402
from shared_utils.llm import CachedChat, truncate_to_tokens, usage  # noqa: E402
from shared_utils.runner import _run_phase  # noqa: E402

logger = logging.getLogger("phase2.answer_prompt_v2")

SPLIT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "dev_holdout.json"
PHASE23_DIR = settings.paths.results_dir / "phase_2" / RUN_NAME
ARM1_DIR = settings.paths.results_dir / "phase_2" / "p2_answer_v2_dev"
ARM2_DIR = settings.paths.results_dir / "phase_2" / "p2_answer_v2_labels_dev"
REPORT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "prompt_v2_dev.json"

ARM1_VERSION = "p2-answer-v2"
ARM2_VERSION = "p2-answer-v2-labels"

# The four instructions approved for prompt v2. No question-type names.
SYSTEM_V2 = (
    "You are an assistant for question-answering tasks. Answer only from the passages in the context.\n"
    "\n"
    "Answer exactly what the question asks. Do not add items, background, or details the question did not ask for.\n"
    "Cover every part of the question. Give names, numbers, dates, and identifiers exactly as written in the passages.\n"
    "If passages disagree, say that they disagree and state which one is current, using dates or version markers when they are present.\n"
    "If the passages do not contain the answer, say so. If they contain only part of it, answer that part and say which part is missing. Do not guess."
)
SYSTEM_V2_LABELS = SYSTEM_V2 + "\nEach passage starts with a label giving its title and source."
FLOOR_LINE = (
    "No passage scored as clearly relevant. The passage below is the closest "
    "match and may not contain the answer."
)

_JEV_RE = re.compile(
    r"refusal=(\w+)\(([0-9.]+)\) main_point=([0-9.]+) contradicts=([0-9.]+)"
)


def user_prompt(question: str, context: str, *, floor: bool) -> str:
    """Phase 2.3's user layout. The floor line is inserted only for that case."""
    note = f"\n{FLOOR_LINE}\n" if floor else ""
    return f"Question: {question}\n{note}\nContext:\n{context}\n\nAnswer:"


def _label(n: int, title: str, source: str) -> str:
    """One line. A title with a line break would split the label, so whitespace is flattened."""
    flat = " ".join(title.split())
    return f"[Passage {n}] Title: {flat} | Source: {source}"


def render_context(passages: list[dict], *, labeled: bool) -> str:
    """Passages in the order given (Jev score, highest first), joined as Phase 2.3 joins them."""
    blocks = []
    for n, passage in enumerate(passages, start=1):
        text = passage["text"]
        if labeled:
            text = _label(n, passage["title"], passage["source"]) + "\n" + text
        blocks.append(text)
    return "\n\n".join(blocks)


def load_split() -> tuple[list[str], list[str]]:
    raw = json.loads(SPLIT_PATH.read_text())
    dev, hold = list(raw["dev"]), list(raw["holdout"])
    if len(dev) != 210 or len(hold) != 210 or set(dev) & set(hold):
        raise RuntimeError(f"DEV/HOLDOUT split is not 210 and 210 disjoint (got {len(dev)} and {len(hold)}).")
    return dev, hold


def prepare(questions, probs, index: DocIndex) -> dict[str, dict]:
    """The Phase 2.3 passages for each question, plus whether the floor kept the only one."""
    prepared = {}
    for q in questions:
        ranked = select_sent(q, probs[q.question_id], threshold=PASS_AT, order="jev")
        prob_of = {id(cand): p for cand, p in zip(q.candidates, probs[q.question_id])}
        passages = []
        for cand, mode, n_tokens in ranked:
            text = passage_text(cand, index)
            if mode.endswith("+cut"):
                text = truncate_to_tokens(text, n_tokens)
            row = index._rows[cand.path]
            passages.append({
                "doc_id": cand.doc_id,
                "text": text,
                "title": row.title,
                "source": row.source_type,
                "prob": prob_of[id(cand)],
            })
        if not passages:
            raise RuntimeError(f"{q.question_id} has an empty prompt.")
        prepared[q.question_id] = {
            "question": q.question,
            "passages": passages,
            "floor": all(p["prob"] < PASS_AT for p in passages),
            "doc_ids": [p["doc_id"] for p in passages],
        }
    return prepared


def _messages(item: dict, *, labeled: bool) -> list[tuple[str, str]]:
    system = SYSTEM_V2_LABELS if labeled else SYSTEM_V2
    context = render_context(item["passages"], labeled=labeled)
    return [("system", system), ("user", user_prompt(item["question"], context, floor=item["floor"]))]


async def _answer_arm(name: str, out_dir: Path, prepared: dict[str, dict], wanted: set[str],
                      *, labeled: bool, version: str, dry_run: bool, only_empty: bool) -> int:
    by_text = {item["question"]: qid for qid, item in prepared.items()}
    if len(by_text) != len(prepared):
        raise RuntimeError("Two questions share the same text; answering keys on the text.")
    llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens, dry_run=dry_run)

    async def answer(question: str) -> dict:
        item = prepared[by_text[question]]
        res = await llm.ainvoke(_messages(item, labeled=labeled), prompt_version=version)
        return {"answer": res.text, "doc_ids": item["doc_ids"]}

    usage.reset()
    args = argparse.Namespace(
        limit=None, question_type=None, question_types=None,
        dry_run=dry_run, evaluate=not dry_run, concurrency=5, only_empty=only_empty)
    code = await _run_phase(name, answer, args, out_dir, "jev", wanted)
    # A real run resets the tracker before judging, and writes the answer bill
    # to answer_cost.json first. A dry run leaves the upper bound in the tracker.
    print(f"\n=== {name} {'dry-run upper bound' if dry_run else 'answer cost'} ===")
    if dry_run:
        print(json.dumps(usage.report(), indent=2))
    else:
        print((out_dir / "answer_cost.json").read_text())
        judge_cost = out_dir / "jev" / "answers_metrics.json"
        if judge_cost.is_file():
            paid = json.loads(judge_cost.read_text()).get("judge_cost_usd")
            print(f"judge cost ${paid}")
    return code


def parse_jev(rationale: str) -> tuple[str, float, float, float] | None:
    match = _JEV_RE.search(rationale or "")
    if not match:
        return None
    choice, _, main_p, contra_p = match.groups()
    return choice, float(main_p), float(contra_p), 0.0


def _flags(row: dict) -> dict:
    parsed = parse_jev(row.get("correctness_rationale") or "")
    if parsed is None:
        raise RuntimeError(f"{row['question_id']} has no Jev rationale.")
    choice, main_p, contra_p, _ = parsed
    return {
        "declines": choice == "declines",
        "contradiction_alone": (
            row["correct"] is False and choice == "answers" and main_p >= 0.7 and contra_p >= 0.3
        ),
    }


def _load_eval(path: Path) -> dict[str, dict]:
    return {row["question_id"]: row for row in map(json.loads, path.open())}


def _scores(rows: list[dict]) -> list[QuestionScore]:
    fields = QuestionScore.__dataclass_fields__
    return [QuestionScore(**{key: row[key] for key in fields}) for row in rows]


def _flip_counts(before: dict[str, dict], after: dict[str, dict], ids: list[str]) -> Counter:
    counts: Counter = Counter()
    for qid in ids:
        old, new = bool(before[qid]["correct"]), bool(after[qid]["correct"])
        if not old and new:
            counts["wrong_to_right"] += 1
        elif old and not new:
            counts["right_to_wrong"] += 1
        elif new:
            counts["both_right"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _share(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return round(100 * sum(_flags(row)[key] for row in rows) / len(rows), 1)


def _metrics(rows: list[dict]) -> dict:
    summary = aggregate(_scores(rows))
    summary["contradiction_alone_pct"] = _share(rows, "contradiction_alone")
    summary["declines_pct"] = _share(rows, "declines")
    return summary


def _examples(qa, old_answers, new_answers, before, after, ids: list[str]) -> dict:
    """Five fixes and five breaks, spread across question types."""
    buckets: dict[str, list[str]] = {"fixed": [], "broke": []}
    by_type: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"fixed": [], "broke": []})
    for qid in ids:
        old, new = bool(before[qid]["correct"]), bool(after[qid]["correct"])
        kind = "fixed" if (not old and new) else "broke" if (old and not new) else ""
        if kind:
            by_type[before[qid]["question_type"]][kind].append(qid)
    for kind in ("fixed", "broke"):
        while len(buckets[kind]) < 5:
            added = False
            for qtype in FOCUS:
                pending = [qid for qid in by_type[qtype][kind] if qid not in buckets[kind]]
                if pending:
                    buckets[kind].append(pending[0])
                    added = True
                if len(buckets[kind]) == 5:
                    break
            if not added:
                break

    def pack(qid: str) -> dict:
        return {
            "question_id": qid,
            "question_type": before[qid]["question_type"],
            "question": qa.loc[qid, "question"],
            "gold_answer": qa.loc[qid, "gold_answer"],
            "phase23_answer": old_answers[qid].get("answer") or "",
            "arm1_answer": new_answers[qid].get("answer") or "",
            "phase23_jev": before[qid].get("correctness_rationale"),
            "arm1_jev": after[qid].get("correctness_rationale"),
        }

    return {kind: [pack(qid) for qid in qids] for kind, qids in buckets.items()}


def report(prepared: dict[str, dict], dev: list[str]) -> dict:
    qa = load_qa()
    phase23 = _load_eval(PHASE23_DIR / "jev" / "answers_eval.jsonl")
    arm1 = _load_eval(ARM1_DIR / "jev" / "answers_eval.jsonl")
    arm2 = _load_eval(ARM2_DIR / "jev" / "answers_eval.jsonl")
    guard = [qid for qid, item in prepared.items() if phase23[qid]["question_type"] in GUARD]
    if len(guard) != 20:
        raise RuntimeError(f"Expected 20 info_not_found questions, got {len(guard)}.")

    def block(name: str, rows_by_id: dict[str, dict], ids: list[str]) -> dict:
        rows = [rows_by_id[qid] for qid in ids]
        summary = _metrics(rows)
        by_type = {}
        for qtype in FOCUS:
            subset = [row for row in rows if row["question_type"] == qtype]
            if subset:
                by_type[qtype] = _metrics(subset)
        return {"name": name, "overall": summary, "by_type": by_type}

    dev_blocks = [
        block("phase23", phase23, dev),
        block("arm1", arm1, dev),
        block("arm2", arm2, dev),
    ]
    flips = {
        "arm1_vs_phase23": dict(_flip_counts(phase23, arm1, dev)),
        "arm2_vs_phase23": dict(_flip_counts(phase23, arm2, dev)),
        "arm2_vs_arm1": dict(_flip_counts(arm1, arm2, dev)),
    }
    flips_by_type = {}
    for qtype in FOCUS:
        ids = [qid for qid in dev if phase23[qid]["question_type"] == qtype]
        flips_by_type[qtype] = {
            "arm1_vs_phase23": dict(_flip_counts(phase23, arm1, ids)),
            "arm2_vs_phase23": dict(_flip_counts(phase23, arm2, ids)),
            "arm2_vs_arm1": dict(_flip_counts(arm1, arm2, ids)),
        }

    floor_ids = [qid for qid in dev if prepared[qid]["floor"]]
    gold_in = gold_missing = no_gold = 0
    for qid in floor_ids:
        gold = dedupe(list(qa.loc[qid, "expected_doc_ids"]))
        sent = set(prepared[qid]["doc_ids"])
        if not gold:
            no_gold += 1
        elif all(doc_id in sent for doc_id in gold):
            gold_in += 1
        else:
            gold_missing += 1
    floor = {
        "n": len(floor_ids),
        "gold_in_prompt": gold_in,
        "gold_missing": gold_missing,
        "no_gold_document": no_gold,
        "phase23": aggregate(_scores([phase23[qid] for qid in floor_ids])).get("overall") if floor_ids else {},
        "arm1": aggregate(_scores([arm1[qid] for qid in floor_ids])).get("overall") if floor_ids else {},
        "arm2": aggregate(_scores([arm2[qid] for qid in floor_ids])).get("overall") if floor_ids else {},
    }
    guard_block = {
        "phase23": aggregate(_scores([phase23[qid] for qid in guard]))["overall"],
        "arm1": aggregate(_scores([arm1[qid] for qid in guard]))["overall"],
        "arm2": aggregate(_scores([arm2[qid] for qid in guard]))["overall"],
        "arm1_flips": dict(_flip_counts(phase23, arm1, guard)),
        "arm2_flips": dict(_flip_counts(phase23, arm2, guard)),
    }
    examples = _examples(
        qa,
        {row["question_id"]: row for row in load_answers(PHASE23_DIR / "answers.jsonl")},
        {row["question_id"]: row for row in load_answers(ARM1_DIR / "answers.jsonl")},
        phase23, arm1, dev)
    out = {
        "dev": dev_blocks,
        "flips": flips,
        "flips_by_type": flips_by_type,
        "floor": floor,
        "info_not_found": guard_block,
        "examples_arm1_vs_phase23": examples,
    }
    REPORT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    _print_report(out)
    return out


def _print_report(out: dict) -> None:
    print("\n=== DEV (210), Jev. Phase 2.3 vs prompt v2 ===")
    print(f"{'':22s}{'overall':>10s}{'correct':>10s}{'complete':>10s}{'words':>8s}{'contra%':>9s}{'declines%':>10s}")
    for block in out["dev"]:
        row = block["overall"]["overall"]
        print(f"{block['name']:22s}{row['overall_score']:10.2f}{row['correctness']:10.2f}"
              f"{row['completeness']:10.2f}{row['avg_answer_words']:8.1f}"
              f"{block['overall']['contradiction_alone_pct']:9.1f}{block['overall']['declines_pct']:10.1f}")
    flips = out["flips"]
    for key in ("arm1_vs_phase23", "arm2_vs_phase23", "arm2_vs_arm1"):
        row = flips[key]
        print(f"{key}: {row.get('wrong_to_right', 0)} wrong->right, {row.get('right_to_wrong', 0)} right->wrong")

    print(f"\n{'type':28s}{'p23':>8s}{'arm1':>8s}{'arm2':>8s}{'a1 w>r':>8s}{'a1 r>w':>8s}{'a2 w>r':>8s}{'a2 r>w':>8s}")
    for qtype in FOCUS:
        cells = []
        for block in out["dev"]:
            cells.append(f"{block['by_type'][qtype]['overall']['overall_score']:8.2f}")
        a1 = out["flips_by_type"][qtype]["arm1_vs_phase23"]
        a2 = out["flips_by_type"][qtype]["arm2_vs_phase23"]
        print(f"{qtype:28s}{''.join(cells)}{a1.get('wrong_to_right', 0):8d}{a1.get('right_to_wrong', 0):8d}"
              f"{a2.get('wrong_to_right', 0):8d}{a2.get('right_to_wrong', 0):8d}")

    floor = out["floor"]
    print(f"\nFloor line on DEV: {floor['n']}. Gold in the prompt: {floor['gold_in_prompt']}. "
          f"Gold missing: {floor['gold_missing']}. No gold document: {floor['no_gold_document']}.")
    print(f"{'':22s}{'overall':>10s}{'correct':>10s}{'complete':>10s}{'words':>8s}")
    for name in ("phase23", "arm1", "arm2"):
        row = floor[name]
        if not row:
            continue
        print(f"{name:22s}{row['overall_score']:10.2f}{row['correctness']:10.2f}"
              f"{row['completeness']:10.2f}{row['avg_answer_words']:8.1f}")

    guard = out["info_not_found"]
    print("\ninfo_not_found (20), against Phase 2.3's 95.00")
    for name in ("phase23", "arm1", "arm2"):
        row = guard[name]
        print(f"{name:22s}{row['overall_score']:10.2f}{row['correctness']:10.2f}"
              f"{row['completeness']:10.2f}{row['avg_answer_words']:8.1f}")


def _print_examples(out: dict) -> None:
    examples = out["examples_arm1_vs_phase23"]
    for kind in ("fixed", "broke"):
        print(f"\n=== ARM 1 {kind} vs Phase 2.3 ===")
        for row in examples[kind]:
            print(f"\n[{row['question_type']}] {row['question_id']}")
            print(f"Q: {' '.join(row['question'].split())}")
            print(f"Gold: {' '.join(row['gold_answer'].split())[:500]}")
            print(f"Phase 2.3 ({row['phase23_jev']}): {' '.join(row['phase23_answer'].split())[:500]}")
            print(f"ARM 1 ({row['arm1_jev']}): {' '.join(row['arm1_answer'].split())[:500]}")


async def main_async(args: argparse.Namespace) -> int:
    dev, holdout = load_split()
    questions, probs = load_fused()
    by_id = {q.question_id: q for q in questions}
    guard = [q.question_id for q in questions if q.question_type in GUARD]
    wanted = set(dev) | set(guard)
    if wanted & set(holdout):
        raise RuntimeError("HOLDOUT ids leaked into the answer set.")
    selected = [by_id[qid] for qid in wanted]
    if len(selected) != 230:
        raise RuntimeError(f"Expected 230 questions (210 DEV + 20 guard), got {len(selected)}.")
    index = DocIndex()
    prepared = prepare(selected, probs, index)
    floor_dev = sum(prepared[qid]["floor"] for qid in dev)
    floor_guard = sum(prepared[qid]["floor"] for qid in guard)
    print(f"Prepared 230 questions. Floor line on {floor_dev} DEV and {floor_guard} info_not_found.")
    sample = next((qid for qid in list(dev) + guard if prepared[qid]["floor"]), guard[0])
    labeled = _messages(prepared[sample], labeled=True)
    print(f"\nSample floor prompt ({sample}), system tail and user head:")
    print(labeled[0][1].splitlines()[-1])
    print("---")
    print("\n".join(labeled[1][1].splitlines()[:8]))

    code = await _answer_arm(
        "arm1", ARM1_DIR, prepared, wanted, labeled=False, version=ARM1_VERSION,
        dry_run=args.dry_run, only_empty=args.only_empty)
    if code != 0:
        return code
    code = await _answer_arm(
        "arm2", ARM2_DIR, prepared, wanted, labeled=True, version=ARM2_VERSION,
        dry_run=args.dry_run, only_empty=args.only_empty)
    if code != 0:
        return code
    if args.dry_run:
        return 0
    out = report(prepared, dev)
    _print_examples(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer DEV and the guard with prompt v2. Not HOLDOUT.")
    parser.add_argument("--dry-run", action="store_true", help="Price both arms and do not call the models.")
    parser.add_argument("--only-empty", action="store_true", help="Re-answer only empty or failed rows.")
    parser.add_argument("--report-only", action="store_true", help="Reprint the report from saved answers.")
    args = parser.parse_args(argv)
    setup_logging()
    if args.report_only:
        dev, _hold = load_split()
        questions, probs = load_fused()
        by_id = {q.question_id: q for q in questions}
        guard = [q.question_id for q in questions if q.question_type in GUARD]
        index = DocIndex()
        prepared = prepare([by_id[qid] for qid in set(dev) | set(guard)], probs, index)
        out = report(prepared, dev)
        _print_examples(out)
        return 0
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.error("Interrupted. Completed calls are cached; re-run to resume.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
