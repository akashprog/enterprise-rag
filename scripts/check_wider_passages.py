"""Two checks on the fused focus answers. No new answers.

    python scripts/check_wider_passages.py --dry-run
        Wider passages ($0) plus a sol correctness dry-run (upper-bound cost).

    python scripts/check_wider_passages.py
        The same passage table, then the sol correctness calls.

Sol sees only correctness, on two sets: the 27 answers whose gold facts were
mostly in the prompt and that Jev still called wrong, and the focus answers
Jev called correct with a contradiction score from 0.20 to 0.30.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_2_data_mastery.hybrid_candidates import (  # noqa: E402
    FLOOR,
    FOCUS,
    PASS_AT,
    RUN_NAME,
    _sent_passages,
    load_fused,
)
from phase_2_data_mastery.query_small_to_big import DocIndex, passage_for  # noqa: E402
from phase_2_data_mastery.relevance_filter import (  # noqa: E402
    DEFAULT_BUDGET,
    estimate_cost,
    prompt_tokens,
    select_by_probability,
)
from scripts.analyze_recall_failures import (  # noqa: E402
    FACT_PRESENT_MIN,
    _REFUSAL_RE,
    content_words,
    fact_coverage,
)
from shared_utils.config import settings  # noqa: E402
from shared_utils.evaluation import Judge, dedupe, load_answers, load_qa, strip_citations  # noqa: E402
from shared_utils.llm import run_limited, truncate_to_tokens, usage  # noqa: E402

NEW_DIR = settings.paths.results_dir / "phase_2" / RUN_NAME
OUT = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "wider_passages.json"
SOL_OUT = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "sol_correctness.json"

# (label, window, whole-document token cap). None means always send the whole document.
SETTINGS = (
    ("a_w1_t2000", 1, 2000),
    ("b_w2_t4000", 2, 4000),
    ("c_w3_t8000", 3, 8000),
    ("d_whole", 0, None),
)

_JEV_RE = re.compile(
    r"refusal=(\w+)\(([0-9.]+)\) main_point=([0-9.]+) contradicts=([0-9.]+)"
)


def parse_jev(rationale: str) -> tuple[str, float, float, float] | None:
    match = _JEV_RE.search(rationale or "")
    if not match:
        return None
    choice, refusal_p, main_p, contra_p = match.groups()
    return choice, float(refusal_p), float(main_p), float(contra_p)


def jev_fail_reason(parsed: tuple[str, float, float, float]) -> str:
    choice, _, main_p, contra_p = parsed
    low_main = main_p < 0.7
    contradicts = contra_p >= 0.3
    if choice != "answers":
        return "declines"
    if contradicts and not low_main:
        return "contradiction_alone"
    if low_main and not contradicts:
        return "main_point"
    if low_main and contradicts:
        return "both"
    return "other"


def pack(q, probs: list[float], index: DocIndex, window: int, whole_max: int | None):
    """Same documents the filter would consider, then the 10k budget on new passage sizes.

    The candidate order is Jev score, as in the answered run. Passage text is the
    only thing that changes. Returns sent passages and how the budget treated
    the documents that passed the score rule.
    """
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    cands = [q.candidates[i] for i in order]
    ranked_probs = [probs[i] for i in order]
    built = []
    fakes = []
    cap = whole_max if whole_max is not None else 10**9
    for cand in cands:
        mode, text, n_tokens = passage_for(
            index.get(cand.path), cand.chunks, window=window, whole_doc_max=cap)
        fake = replace(cand, mode=mode, passage_tokens=n_tokens)
        built.append((fake, text))
        fakes.append(fake)
    sent_spec = select_by_probability(
        fakes, ranked_probs, threshold=PASS_AT, budget=DEFAULT_BUDGET, floor=FLOOR)
    text_of = {id(fake): text for fake, text in built}
    sent = []
    for cand, mode, n_tokens in sent_spec:
        text = text_of[id(cand)]
        if mode.endswith("+cut"):
            text = truncate_to_tokens(text, n_tokens)
        sent.append({"doc_id": cand.doc_id, "mode": mode, "tokens": n_tokens, "text": text})

    # Documents the score rule wanted, in score order, before the budget.
    chosen = [i for i, p in enumerate(ranked_probs) if p >= PASS_AT]
    if len(chosen) < FLOOR:
        for i in range(len(fakes)):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= FLOOR:
                break
    chosen.sort()
    sent_ids = {row["doc_id"] for row in sent}
    dropped = [fakes[i].doc_id for i in chosen if fakes[i].doc_id not in sent_ids]
    truncated = [row["doc_id"] for row in sent if row["mode"].endswith("+cut")]
    return sent, dropped, truncated


def fact_hits(facts: list[str], text: str) -> tuple[int, list[str]]:
    words = content_words(text)
    present = [fact for fact in facts if fact_coverage(fact, words) >= FACT_PRESENT_MIN]
    return len(present), present


def passage_report(questions, probs, index, qa, new_eval, new_answers) -> dict:
    focus = [q for q in questions if q.question_type in FOCUS]
    by_id = {q.question_id: q for q in focus}

    # Current prompt, to name the 74 and the 10 and to check the rebuild.
    current_text: dict[str, str] = {}
    current_ids: dict[str, list[str]] = {}
    for q in focus:
        texts, doc_ids = _sent_passages(q, probs[q.question_id], index)
        current_text[q.question_id] = "\n\n".join(texts)
        current_ids[q.question_id] = doc_ids
        saved = list(new_answers[q.question_id]["doc_ids"])
        if doc_ids != saved:
            raise RuntimeError(f"{q.question_id}: rebuilt docs differ from the saved answer.")

    incomplete = []
    absent_facts: list[tuple[str, str]] = []
    under_half = []
    for q in focus:
        ev = new_eval[q.question_id]
        facts = list(qa.loc[q.question_id, "answer_facts"])
        present, _missing = fact_hits(facts, current_text[q.question_id])
        gold = dedupe(list(qa.loc[q.question_id, "expected_doc_ids"]))
        gold_missing = any(doc_id not in current_ids[q.question_id] for doc_id in gold)
        share = present / len(facts) if facts else None
        if ev["correct"] is True and ev["completeness"] is not None and ev["completeness"] < 1:
            unsupported = list(ev.get("unsupported_facts") or [])
            words = content_words(current_text[q.question_id])
            for fact in unsupported:
                if fact_coverage(fact, words) < FACT_PRESENT_MIN:
                    absent_facts.append((q.question_id, fact))
            incomplete.append(q.question_id)
        # Same order as the diagnosis: a missing gold document is not this group,
        # and a question with no gold documents can still land here.
        if ev["correct"] is False and not gold_missing and share is not None and share < 0.5:
            under_half.append(q.question_id)
    if len(incomplete) != 74:
        raise RuntimeError(f"Expected 74 correct-but-incomplete answers, got {len(incomplete)}.")
    if len(absent_facts) != 125:
        raise RuntimeError(f"Expected 125 facts outside the prompt, got {len(absent_facts)}.")
    if len(under_half) != 10:
        raise RuntimeError(f"Expected 10 under-half answers, got {len(under_half)}.")

    # How the current rule sent each incomplete question's gold documents.
    gold_shape = Counter()
    for qid in incomplete:
        q = by_id[qid]
        sent, _dropped, _truncated = pack(q, probs[qid], index, window=1, whole_max=2000)
        modes = {row["doc_id"]: row["mode"].replace("+cut", "") for row in sent}
        gold = dedupe(list(qa.loc[qid, "expected_doc_ids"]))
        if not gold:
            gold_shape["no_gold_document"] += 1
            continue
        present_modes = [modes[doc_id] for doc_id in gold if doc_id in modes]
        missing = len(gold) - len(present_modes)
        if missing and not present_modes:
            gold_shape["gold_not_in_prompt"] += 1
        elif missing:
            gold_shape["some_gold_missing"] += 1
        elif all(mode == "whole" for mode in present_modes):
            gold_shape["whole_document"] += 1
        elif all(mode == "window" for mode in present_modes):
            gold_shape["window"] += 1
        else:
            gold_shape["mixed"] += 1

    rows = []
    for label, window, whole_max in SETTINGS:
        fact_pool = Counter()
        shares: dict[str, list[float]] = {"focus": [], "incomplete": [], "under_half": []}
        prompt_total = 0
        truncated = dropped = 0
        recovered = 0
        for q in focus:
            sent, drop, cuts = pack(q, probs[q.question_id], index, window, whole_max)
            truncated += len(cuts)
            dropped += len(drop)
            text = "\n\n".join(row["text"] for row in sent)
            prompt_total += prompt_tokens(q.question, [(row["text"], row["tokens"]) for row in sent])
            facts = list(qa.loc[q.question_id, "answer_facts"])
            present, _ = fact_hits(facts, text)
            if facts:
                share = present / len(facts)
                shares["focus"].append(share)
                fact_pool["focus_present"] += present
                fact_pool["focus_total"] += len(facts)
                if q.question_id in incomplete:
                    shares["incomplete"].append(share)
                    fact_pool["incomplete_present"] += present
                    fact_pool["incomplete_total"] += len(facts)
                if q.question_id in under_half:
                    shares["under_half"].append(share)
                    fact_pool["under_present"] += present
                    fact_pool["under_total"] += len(facts)
            if label != "a_w1_t2000":
                words = content_words(text)
                recovered += sum(fact_coverage(fact, words) >= FACT_PRESENT_MIN
                                 for qid, fact in absent_facts if qid == q.question_id)
        n = len(focus)
        def pooled(name: str) -> float | None:
            total = fact_pool[f"{name}_total"]
            return None if not total else round(fact_pool[f"{name}_present"] / total, 4)

        rows.append({
            "setting": label,
            "window": window,
            "whole_doc_max": whole_max,
            "fact_share": {
                "focus": pooled("focus"),
                "correct_but_incomplete": pooled("incomplete"),
                "under_half": pooled("under"),
            },
            "questions": {name: len(values) for name, values in shares.items()},
            "facts_now_inside_of_125": 0 if label == "a_w1_t2000" else recovered,
            "prompt_tokens_total": prompt_total,
            "prompt_tokens_avg": round(prompt_total / n),
            "documents_truncated": truncated,
            "documents_left_out": dropped,
            "est_answer_cost_usd": estimate_cost(prompt_total, n),
        })
        # Setting (a) must be the prompt that produced the 125.
        if label == "a_w1_t2000":
            for q in focus:
                sent, _, _ = pack(q, probs[q.question_id], index, window, whole_max)
                if [row["doc_id"] for row in sent] != current_ids[q.question_id]:
                    raise RuntimeError(f"{q.question_id}: setting (a) docs differ from the answered prompt.")

    return {
        "incomplete_n": len(incomplete),
        "under_half_n": len(under_half),
        "facts_outside_prompt": len(absent_facts),
        "gold_shape_of_incomplete": dict(gold_shape),
        "settings": rows,
    }


def judge_sets(qa, new_eval, new_answers) -> list[dict]:
    """The 27 still-wrong answers, and correct answers with contradiction in [0.20, 0.30]."""
    items = []
    for qid, ev in new_eval.items():
        if ev["question_type"] not in FOCUS:
            continue
        parsed = parse_jev(ev.get("correctness_rationale") or "")
        if parsed is None:
            raise RuntimeError(f"{qid} has no Jev scores in its rationale.")
        answer = strip_citations(new_answers[qid].get("answer") or "")
        row = qa.loc[qid]
        base = {
            "question_id": qid,
            "question_type": ev["question_type"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "answer": answer,
            "jev_rationale": ev.get("correctness_rationale"),
            "jev_correct": ev["correct"],
            "jev_main_point": parsed[2],
            "jev_contradicts": parsed[3],
        }
        if ev["correct"] is False:
            # Group (d) is applied by the caller; this function only parses.
            base["why"] = jev_fail_reason(parsed)
            items.append(base)
        elif 0.20 <= parsed[3] <= 0.30:
            base["why"] = "correct_contradiction_band"
            base["set"] = "correct_band"
            items.append(base)
    return items


def classify_wrong(questions, probs, index, qa, new_eval, new_answers) -> list[str]:
    """Question ids in the 'facts present, still wrong' group. Must be 27."""
    ids = []
    focus = {q.question_id: q for q in questions if q.question_type in FOCUS}
    for qid, q in focus.items():
        ev = new_eval[qid]
        if ev["correct"] is not False:
            continue
        texts, doc_ids = _sent_passages(q, probs[qid], index)
        gold = dedupe(list(qa.loc[qid, "expected_doc_ids"]))
        if any(doc_id not in doc_ids for doc_id in gold):
            continue
        facts = list(qa.loc[qid, "answer_facts"])
        present, _ = fact_hits(facts, "\n\n".join(texts))
        share = present / len(facts) if facts else None
        if share is not None and share < 0.5:
            continue
        answer = new_answers[qid].get("answer") or ""
        if _REFUSAL_RE.search(answer):
            continue
        ids.append(qid)
    if len(ids) != 27:
        raise RuntimeError(f"Expected 27 facts-present-still-wrong answers, got {len(ids)}.")
    return ids


async def judge(items: list[dict], *, dry_run: bool) -> list[dict]:
    usage.reset()
    judge_ = Judge(mode="sol", dry_run=dry_run)

    async def one(item: dict) -> dict:
        correct, rationale, who = await judge_.correctness(
            item["question"], item["gold_answer"], item["answer"], item["question_type"])
        return {**item, "sol_correct": correct, "sol_rationale": rationale, "sol_judge": who}

    results = await run_limited(items, one, limit=settings.max_concurrency, desc="sol correctness")
    await judge_.aclose()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Price the sol calls; do not send them.")
    parser.add_argument("--passages-only", action="store_true", help="Skip the sol judge.")
    args = parser.parse_args()

    qa = load_qa()
    new_eval = {r["question_id"]: r for r in map(json.loads, (NEW_DIR / "jev" / "answers_eval.jsonl").open())}
    new_answers = {a["question_id"]: a for a in load_answers(NEW_DIR / "answers.jsonl")}
    questions, probs = load_fused()
    index = DocIndex()

    report = passage_report(questions, probs, index, qa, new_eval, new_answers)
    wrong_ids = set(classify_wrong(questions, probs, index, qa, new_eval, new_answers))
    parsed_rows = judge_sets(qa, new_eval, new_answers)
    still_wrong = [row for row in parsed_rows if row["question_id"] in wrong_ids]
    # judge_sets also appended every wrong answer. Keep only the 27, plus the band.
    band = [row for row in parsed_rows if row.get("set") == "correct_band"]
    for row in still_wrong:
        row["set"] = "facts_present_still_wrong"
    if len(still_wrong) != 27:
        raise RuntimeError(f"Judge set is {len(still_wrong)}, expected 27.")
    items = still_wrong + band

    print("\n=== Wider passages ($0) ===")
    print("gold shape of the 74:", report["gold_shape_of_incomplete"])
    for row in report["settings"]:
        share = row["fact_share"]
        print(f"{row['setting']:12s} facts focus {share['focus']:.3f}  "
              f"incomplete {share['correct_but_incomplete']:.3f}  under-half {share['under_half']:.3f}  "
              f"of125 {row['facts_now_inside_of_125']:3d}  "
              f"prompt avg {row['prompt_tokens_avg']:5d}  total {row['prompt_tokens_total']:8d}  "
              f"truncated {row['documents_truncated']:3d}  left out {row['documents_left_out']:3d}  "
              f"est ${row['est_answer_cost_usd']}")
    OUT.write_text(json.dumps(report, indent=2))

    print(f"\n=== Sol correctness {'dry-run' if args.dry_run else 'run'} ===")
    print(f"facts-present-still-wrong: {len(still_wrong)}  "
          f"reasons {Counter(row['why'] for row in still_wrong)}")
    print(f"correct with contradiction 0.20-0.30: {len(band)}")
    if args.passages_only:
        return 0

    results = asyncio.run(judge(items, dry_run=args.dry_run))
    cost = usage.report()
    print(json.dumps(cost, indent=2))
    if args.dry_run:
        SOL_OUT.write_text(json.dumps({"dry_run": cost, "n": len(items)}, indent=2))
        return 0

    disagreements = [row for row in results if row["sol_correct"] is not row["jev_correct"]]
    by_set = {}
    for name in ("facts_present_still_wrong", "correct_band"):
        group = [row for row in results if row["set"] == name]
        sol_correct = sum(row["sol_correct"] is True for row in group)
        by_why = {}
        for why in sorted({row["why"] for row in group}):
            subset = [row for row in group if row["why"] == why]
            by_why[why] = {
                "n": len(subset),
                "sol_correct": sum(row["sol_correct"] is True for row in subset),
            }
        by_set[name] = {"n": len(group), "sol_correct": sol_correct, "by_why": by_why}
        print(f"{name}: sol correct {sol_correct} of {len(group)}  {by_why}")
    print(f"disagreements: {len(disagreements)}")
    SOL_OUT.write_text(json.dumps({"cost": cost, "summary": by_set, "results": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
