"""Parked-question split, empty-answer repair, and the token-budget check.

The split is seed 42, stratified by work group. Tuning stays on DEV.
The budget check rebuilds prompts only. It does not answer.

Usage:
    python -m phase_2_data_mastery.parked_next --split
    python -m phase_2_data_mastery.parked_next --repair-empty
    python -m phase_2_data_mastery.parked_next --budget
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.query_naive import SYSTEM_PROMPT  # noqa: E402
from phase_2_data_mastery.answer_prompt_v2 import prepare  # noqa: E402
from phase_2_data_mastery.floor_line import PROMPT_VERSION as FLOOR_VERSION, user_prompt  # noqa: E402
from phase_2_data_mastery.hybrid_candidates import ANSWER_PROMPT_VERSION, PASS_AT  # noqa: E402
from phase_2_data_mastery.parked_baseline import (  # noqa: E402
    OUT_DIR,
    PARKED,
    SCORES_PATH,
    WORK,
    _eligible,
    _fuse,
    _judge,
    _load_eval,
    _parked,
    _prompt_tokens,
)
from phase_2_data_mastery.query_small_to_big import DocIndex, messages  # noqa: E402
from phase_2_data_mastery.relevance_filter import (  # noqa: E402
    passage_text,
    select_by_probability,
    truncate_to_tokens,
)
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import aggregate, load_answers, retrieval_metrics, save_answers  # noqa: E402
from shared_utils.llm import CachedChat, count_tokens, usage  # noqa: E402

SPLIT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "parked_dev_holdout.json"
BUDGET_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "parked_budget_dev.json"
GROUPS = ("multi-part", "count/list")
# No-limit is larger than any prompt we will build. The filter then drops nothing.
NO_LIMIT = 10**9


def write_split() -> dict:
    """24/24 multi-part and 6/6 count/list. One Random(42), multi-part first."""
    rng = random.Random(42)
    dev: list[str] = []
    hold: list[str] = []
    by_group: dict[str, dict[str, list[str]]] = {}
    for name in GROUPS:
        ids = [qid for qid, group in WORK.items() if group == name]
        rng.shuffle(ids)
        left: list[str] = []
        right: list[str] = []
        for qid in ids:
            if len(left) <= len(right):
                left.append(qid)
            else:
                right.append(qid)
        dev.extend(left)
        hold.extend(right)
        by_group[name] = {"dev": left, "holdout": right}
    if len(dev) != 30 or len(hold) != 30 or set(dev) & set(hold) or set(dev) | set(hold) != set(WORK):
        raise RuntimeError("Parked split is not 30 and 30 covering every tagged question.")
    if len(by_group["multi-part"]["dev"]) != 24 or len(by_group["count/list"]["dev"]) != 6:
        raise RuntimeError("Work-group sizes are not 24/24 and 6/6.")
    body = {
        "seed": 42,
        "rule": (
            "Within each work group, shuffle question ids with random.Random(42). "
            "Multi-part is shuffled first, then count/list, on one generator. "
            "The odd leftover goes to the half that is currently smaller; on a tie it goes to DEV. "
            "Tuning on the parked questions uses DEV only."
        ),
        "dev": dev,
        "holdout": hold,
        "by_group": by_group,
    }
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(body, indent=2))
    return body


def load_parked_split() -> tuple[list[str], list[str]]:
    if not SPLIT_PATH.is_file():
        write_split()
    raw = json.loads(SPLIT_PATH.read_text())
    return list(raw["dev"]), list(raw["holdout"])


def _halves() -> dict:
    dev, hold = load_parked_split()
    cascade = _load_eval(OUT_DIR / "cascade" / "answers_eval.jsonl")
    if (set(dev) | set(hold)) - set(cascade):
        raise RuntimeError("Cascade baseline is missing a parked question.")

    def pack(ids: list[str]) -> dict:
        scored = aggregate([cascade[qid] for qid in ids])
        return {
            "overall": scored["overall"],
            "by_work": {
                name: aggregate([cascade[qid] for qid in ids if WORK[qid] == name])["overall"]
                for name in GROUPS
            },
            "by_type": {
                name: aggregate([cascade[qid] for qid in ids if cascade[qid].question_type == name])["overall"]
                for name in PARKED
            },
        }

    return {"dev": pack(dev), "holdout": pack(hold)}


def _print_half(label: str, block: dict) -> None:
    row = block["overall"]
    print(f"\n{label}  n={row['n']}")
    print(f"  cascade {row['overall_score']:.2f} / {row['correctness']:.2f} / {row['completeness']:.2f}"
          f"   recall {row['recall_at_k']:.2f}   words {row['avg_answer_words']:.1f}")
    for name, part in block["by_work"].items():
        print(f"  {name:12s} n={part['n']:2d}  {part['overall_score']:.2f} / {part['correctness']:.2f} / "
              f"{part['completeness']:.2f}   recall {part['recall_at_k']:.2f}   words {part['avg_answer_words']:.1f}")


def _empty_ids() -> list[str]:
    rows = load_answers(OUT_DIR / "answers.jsonl")
    return [row["question_id"] for row in rows if not (row.get("answer") or "").strip() or row.get("error")]


async def repair_empty() -> list[str]:
    """Re-answer empty rows. The chat client retries them at 8,000 tokens."""
    empty = _empty_ids()
    print(f"Empty parked answers: {empty or 'none'}")
    if not empty:
        return []
    questions = _fuse(_parked())
    raw = {row["question_id"]: row for row in map(json.loads, SCORES_PATH.open())}
    probs = {}
    for q in questions:
        row = raw[q.question_id]
        if row["doc_ids"] != [c.doc_id for c in q.candidates]:
            raise RuntimeError(f"{q.question_id}: saved scores do not match the fused list.")
        probs[q.question_id] = row["probs"]
    index = DocIndex()
    prepared = prepare(questions, probs, index)
    wanted = set(empty)
    llm = CachedChat(settings.answer_model, max_tokens=settings.answer_max_tokens)

    async def one(qid: str) -> dict:
        item = prepared[qid]
        context = "\n\n".join(p["text"] for p in item["passages"])
        if item["floor"]:
            pairs = [("system", SYSTEM_PROMPT), ("user", user_prompt(item["question"], context))]
            version = FLOOR_VERSION
        else:
            pairs = messages(item["question"], context)
            version = ANSWER_PROMPT_VERSION
        res = await llm.ainvoke(pairs, prompt_version=version)
        if not (res.text or "").strip():
            raise RuntimeError(f"{qid} is still empty after the higher cap.")
        return {"question_id": qid, "answer": res.text, "doc_ids": item["doc_ids"]}

    usage.reset()
    fresh = {row["question_id"]: row for row in await asyncio.gather(*(one(qid) for qid in empty))}
    print("\n=== empty-answer retry cost ===")
    print(json.dumps(usage.report(), indent=2))
    merged = []
    for row in load_answers(OUT_DIR / "answers.jsonl"):
        merged.append(fresh.get(row["question_id"], row))
    save_answers(merged, OUT_DIR / "answers.jsonl")
    print("\n=== re-judging the parked baseline with cascade, then Jev ===")
    code = await _judge("cascade", dry_run=False)
    if code != 0:
        raise RuntimeError("Cascade re-judge failed.")
    code = await _judge("jev", dry_run=False)
    if code != 0:
        raise RuntimeError("Jev re-judge failed.")
    return empty


def _sent(q, probs: list[float], budget: int):
    cands = list(q.candidates)
    ps = list(probs)
    order = sorted(range(len(ps)), key=lambda i: -ps[i])
    cands = [cands[i] for i in order]
    ps = [ps[i] for i in order]
    return select_by_probability(cands, ps, threshold=PASS_AT, budget=budget, floor=1)


def budget_check() -> dict:
    """Prompt sizes for the 24 multi-part DEV questions. No model calls."""
    dev, _ = load_parked_split()
    wanted = [qid for qid in dev if WORK[qid] == "multi-part"]
    if len(wanted) != 24:
        raise RuntimeError(f"Expected 24 multi-part DEV questions, got {len(wanted)}.")
    questions = [q for q in _fuse(_parked()) if q.question_id in set(wanted)]
    if len(questions) != 24:
        raise RuntimeError("Fused list is missing a multi-part DEV question.")
    raw = {row["question_id"]: row for row in map(json.loads, SCORES_PATH.open())}
    index = DocIndex()
    answers = {row["question_id"]: row["answer"] for row in load_answers(OUT_DIR / "answers.jsonl")}
    out_tokens = [count_tokens(answers[qid]) for qid in wanted if (answers.get(qid) or "").strip()]
    mean_out = sum(out_tokens) / len(out_tokens)
    price = settings.model_prices.get(settings.answer_model)
    budgets = (("10k", 10_000), ("20k", 20_000), ("30k", 30_000), ("no_limit", NO_LIMIT))
    report = {"questions": 24, "output_tokens_held_fixed": round(mean_out, 1), "budgets": {}}
    for name, budget in budgets:
        gold_found = gold_asked = 0
        recalls = []
        kept = []
        cut = []
        tokens = []
        nongold = []
        for q in questions:
            row = raw[q.question_id]
            if row["doc_ids"] != [c.doc_id for c in q.candidates]:
                raise RuntimeError(f"{q.question_id}: scores do not match the fused list.")
            probs = row["probs"]
            eligible = _eligible(q, probs)
            sent = _sent(q, probs, budget)
            sent_ids = [c.doc_id for c, _, _ in sent]
            if any(doc_id not in eligible for doc_id in sent_ids):
                raise RuntimeError(f"{q.question_id}: sent a document the filter had dropped.")
            gold = list(dict.fromkeys(q.gold))
            gold_found += sum(doc_id in sent_ids for doc_id in gold)
            gold_asked += len(gold)
            recall, _ = retrieval_metrics(sent_ids, gold, settings.recall_k)
            recalls.append(recall)
            kept.append(len(sent_ids))
            cut.append(len([doc_id for doc_id in eligible if doc_id not in sent_ids]))
            nongold.append(sum(doc_id not in set(gold) for doc_id in sent_ids))
            prob_of = {c.doc_id: p for c, p in zip(q.candidates, probs)}
            passages = []
            for cand, mode, n_tokens in sent:
                text = passage_text(cand, index)
                if mode.endswith("+cut"):
                    text = truncate_to_tokens(text, n_tokens)
                passages.append({"text": text, "prob": prob_of[cand.doc_id]})
            tokens.append(_prompt_tokens({
                "question": q.question,
                "passages": passages,
                "floor": all(prob_of[doc_id] < PASS_AT for doc_id in sent_ids),
            }))
        total_in = sum(tokens)
        estimate = None
        upper = None
        if price is not None:
            estimate = round(total_in * price.input / 1e6 + 24 * mean_out * price.output / 1e6, 4)
            upper = round(total_in * price.input / 1e6 + 24 * settings.answer_max_tokens * price.output / 1e6, 4)
        report["budgets"][name] = {
            "budget": None if budget == NO_LIMIT else budget,
            "gold_in_prompt": gold_found,
            "gold_asked": gold_asked,
            "gold_in_prompt_pct": round(100 * gold_found / gold_asked, 2),
            "recall": round(100 * sum(recalls) / len(recalls), 2),
            "docs_kept": round(sum(kept) / len(kept), 2),
            "docs_cut": round(sum(cut) / len(cut), 2),
            "docs_cut_total": sum(cut),
            "prompt_tokens_avg": round(sum(tokens) / len(tokens), 1),
            "prompt_tokens_max": max(tokens),
            "nongold_docs": round(sum(nongold) / len(nongold), 2),
            "estimated_answer_cost_usd": estimate,
            "answer_cost_upper_bound_usd": upper,
        }
    BUDGET_PATH.write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parked split, empty-answer repair, budget check.")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--repair-empty", action="store_true")
    parser.add_argument("--budget", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    if args.split or not (args.repair_empty or args.budget):
        body = write_split()
        print(f"Saved {SPLIT_PATH}")
        print(f"DEV {len(body['dev'])}  HOLDOUT {len(body['holdout'])}")
        for name, part in body["by_group"].items():
            print(f"  {name}: {len(part['dev'])} / {len(part['holdout'])}")
    if args.repair_empty:
        asyncio.run(repair_empty())
    if args.split or args.repair_empty:
        halves = _halves()
        print("\n=== cascade baseline by half ===")
        _print_half("DEV", halves["dev"])
        _print_half("HOLDOUT", halves["holdout"])
        print(json.dumps(halves))
    if args.budget:
        report = budget_check()
        print("\n=== multi-part DEV token budgets. No answers. ===")
        print(f"Output held at the current answers' average, {report['output_tokens_held_fixed']} tokens.")
        print(f"{'budget':10s}{'gold':>12s}{'recall':>8s}{'kept':>8s}{'cut':>8s}{'tok avg':>10s}{'tok max':>10s}"
              f"{'nongold':>9s}{'est $':>8s}{'cap $':>8s}")
        for name, row in report["budgets"].items():
            print(f"{name:10s}{row['gold_in_prompt']:4d}/{row['gold_asked']:<7d}{row['recall']:8.2f}"
                  f"{row['docs_kept']:8.2f}{row['docs_cut']:8.2f}{row['prompt_tokens_avg']:10.1f}"
                  f"{row['prompt_tokens_max']:10d}{row['nongold_docs']:9.2f}"
                  f"{row['estimated_answer_cost_usd']:8.4f}{row['answer_cost_upper_bound_usd']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
