"""Free diagnosis of the fused-top-30 answers on the focus set. No model calls.

Groups the wrong answers, compares the 20 right-to-wrong flips with Phase 2.2's
prompts, and checks whether incomplete-but-correct answers were missing facts
that the prompt already contained.

Usage:
    python scripts/diagnose_fused_focus.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_1_baseline.ingest_naive import load_docs  # noqa: E402
from phase_2_data_mastery.hybrid_candidates import (  # noqa: E402
    FOCUS,
    GUARD,
    PHASE_22_EVAL,
    RUN_NAME,
    _sent_passages,
    load_fused,
)
from phase_2_data_mastery.query_small_to_big import DocIndex  # noqa: E402
from scripts.analyze_recall_failures import (  # noqa: E402
    FACT_PRESENT_MIN,
    _REFUSAL_RE,
    content_words,
    fact_coverage,
)
from shared_utils.config import settings  # noqa: E402
from shared_utils.evaluation import dedupe, load_answers, load_qa  # noqa: E402

P22_ANSWERS = settings.paths.results_dir / "phase_2" / "p07_f1_k30_b10000" / "answers.jsonl"
NEW_DIR = settings.paths.results_dir / "phase_2" / RUN_NAME
OUT = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "diagnosis.json"

GROUPS = (
    "gold_missing",
    "facts_under_half",
    "refuses",
    "facts_present_still_wrong",
)


def _facts_present(facts: list[str], text: str) -> tuple[int, list[str]]:
    words = content_words(text)
    missing = [fact for fact in facts if fact_coverage(fact, words) < FACT_PRESENT_MIN]
    return len(facts) - len(missing), missing


def _clip(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + " ..."


def main() -> int:
    qa = load_qa()
    new_eval = {r["question_id"]: r for r in map(json.loads, (NEW_DIR / "jev" / "answers_eval.jsonl").open())}
    old_eval = {r["question_id"]: r for r in map(json.loads, PHASE_22_EVAL.open())}
    new_answers = {a["question_id"]: a for a in load_answers(NEW_DIR / "answers.jsonl")}
    old_answers = {a["question_id"]: a for a in load_answers(P22_ANSWERS)}
    questions, probs = load_fused()
    by_id = {q.question_id: q for q in questions}
    index = DocIndex()
    prompts: dict[str, tuple[list[str], list[str]]] = {}
    for q in questions:
        if q.question_type in FOCUS or q.question_type in GUARD:
            prompts[q.question_id] = _sent_passages(q, probs[q.question_id], index)

    docs = load_docs(None)
    selection = dict(zip(docs["doc_id"], docs["selection"]))
    titles = dict(zip(docs["doc_id"], docs["title"]))
    evidence_for: dict[str, set[str]] = {}
    for doc_id, raw in zip(docs["doc_id"], docs["distractor_for"]):
        if selection.get(doc_id) != "high_level_evidence" or raw is None or raw != raw:
            continue
        if isinstance(raw, str):
            linked = [part.strip(" '\"") for part in raw.strip("[]").split(",") if part.strip()]
        else:
            linked = list(raw)
        evidence_for[doc_id] = set(linked)

    wrong = []
    for qid, ev in new_eval.items():
        if ev["question_type"] not in FOCUS or ev["correct"] is not False:
            continue
        q = qa.loc[qid]
        texts, doc_ids = prompts[qid]
        if doc_ids != list(new_answers[qid]["doc_ids"]):
            raise RuntimeError(f"{qid}: rebuilt prompt docs differ from the saved answer.")
        gold = dedupe(list(q["expected_doc_ids"]))
        gold_in = [doc_id for doc_id in gold if doc_id in doc_ids]
        facts = list(q["answer_facts"])
        present, missing = _facts_present(facts, "\n\n".join(texts))
        share = (present / len(facts)) if facts else None
        answer = new_answers[qid].get("answer") or ""
        refusal = _REFUSAL_RE.search(answer)
        if any(doc_id not in doc_ids for doc_id in gold):
            group = "gold_missing"
        elif share is not None and share < 0.5:
            group = "facts_under_half"
        elif refusal:
            group = "refuses"
        else:
            group = "facts_present_still_wrong"
        wrong.append({
            "question_id": qid,
            "question_type": ev["question_type"],
            "group": group,
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "answer": answer,
            "jev": ev.get("correctness_rationale"),
            "completeness": ev.get("completeness"),
            "n_gold": len(gold),
            "gold_in_prompt": len(gold_in),
            "n_facts": len(facts),
            "facts_present": present,
            "fact_share": None if share is None else round(share, 3),
            "refusal_phrase": refusal.group(0) if refusal else "",
            "missing_facts": missing,
        })
    if len(wrong) != 59:
        raise RuntimeError(f"Expected 59 wrong focus answers, got {len(wrong)}.")

    by_group: dict[str, list] = {name: [] for name in GROUPS}
    for row in wrong:
        by_group[row["group"]].append(row)
    counts = {name: len(rows) for name, rows in by_group.items()}
    by_type: dict[str, Counter] = defaultdict(Counter)
    for row in wrong:
        by_type[row["question_type"]][row["group"]] += 1

    flips = []
    for qid, ev in new_eval.items():
        if ev["question_type"] not in FOCUS:
            continue
        if old_eval[qid]["correct"] is True and ev["correct"] is False:
            old_ids = list(old_answers[qid]["doc_ids"])
            new_ids = list(new_answers[qid]["doc_ids"])
            gold = dedupe(list(qa.loc[qid, "expected_doc_ids"]))
            # Gold is this question's expected docs. Supporting is a high-level
            # evidence doc tagged for this question. Another question's gold
            # document is neither.
            lost = [
                doc_id for doc_id in dict.fromkeys(old_ids)
                if doc_id not in new_ids and (doc_id in gold or qid in evidence_for.get(doc_id, ()))
            ]
            flips.append({
                "question_id": qid,
                "question_type": ev["question_type"],
                "lost_supporting": [
                    {"doc_id": doc_id,
                     "role": "gold" if doc_id in gold else "high_level_evidence",
                     "title": titles.get(doc_id)}
                    for doc_id in lost
                ],
                "gold_in_phase22": sum(doc_id in old_ids for doc_id in gold),
                "gold_in_fused": sum(doc_id in new_ids for doc_id in gold),
                "n_gold": len(gold),
                "same_documents": set(old_ids) == set(new_ids),
                "same_order": old_ids == new_ids,
                "old_docs": old_ids,
                "new_docs": new_ids,
            })
    if len(flips) != 20:
        raise RuntimeError(f"Expected 20 right-to-wrong flips, got {len(flips)}.")

    conflicting = []
    for qid, ev in new_eval.items():
        if ev["question_type"] != "conflicting_info":
            continue
        gold = dedupe(list(qa.loc[qid, "expected_doc_ids"]))
        old_ids = set(old_answers[qid]["doc_ids"])
        new_ids = set(new_answers[qid]["doc_ids"])
        conflicting.append({
            "question_id": qid,
            "gold_needed": len(gold),
            "in_phase22": sum(doc_id in old_ids for doc_id in gold),
            "in_fused": sum(doc_id in new_ids for doc_id in gold),
            "correct": ev["correct"],
        })
    if len(conflicting) != 20:
        raise RuntimeError(f"Expected 20 conflicting_info questions, got {len(conflicting)}.")

    guard_flip = None
    for qid, ev in new_eval.items():
        if ev["question_type"] not in GUARD:
            continue
        if old_eval[qid]["correct"] is True and ev["correct"] is False:
            texts, doc_ids = prompts[qid]
            q = by_id[qid]
            kept = []
            for doc_id, text in zip(doc_ids, texts):
                score = next(p for cand, p in zip(q.candidates, probs[qid]) if cand.doc_id == doc_id)
                kept.append({"doc_id": doc_id, "title": titles.get(doc_id), "jev_score": score,
                             "passage": text})
            guard_flip = {
                "question_id": qid,
                "question": qa.loc[qid, "question"],
                "gold_answer": qa.loc[qid, "gold_answer"],
                "phase22_answer": old_answers[qid].get("answer") or "",
                "fused_answer": new_answers[qid].get("answer") or "",
                "phase22_jev": old_eval[qid].get("correctness_rationale"),
                "fused_jev": ev.get("correctness_rationale"),
                "kept": kept,
            }
    if guard_flip is None:
        raise RuntimeError("The info_not_found right-to-wrong flip was not found.")

    incomplete = []
    fact_in = fact_out = 0
    answer_split = Counter()
    for qid, ev in new_eval.items():
        if ev["question_type"] not in FOCUS or ev["correct"] is not True:
            continue
        if ev["completeness"] is None or ev["completeness"] >= 1:
            continue
        missing = list(ev.get("unsupported_facts") or [])
        text = "\n\n".join(prompts[qid][0])
        words = content_words(text)
        in_prompt = [fact for fact in missing if fact_coverage(fact, words) >= FACT_PRESENT_MIN]
        absent = [fact for fact in missing if fact not in in_prompt]
        fact_in += len(in_prompt)
        fact_out += len(absent)
        if missing and not absent:
            kind = "all_missing_facts_in_prompt"
        elif missing and not in_prompt:
            kind = "no_missing_fact_in_prompt"
        elif missing:
            kind = "mixed"
        else:
            kind = "no_unsupported_list"
        answer_split[kind] += 1
        incomplete.append({
            "question_id": qid,
            "question_type": ev["question_type"],
            "completeness": ev["completeness"],
            "missing_in_prompt": len(in_prompt),
            "missing_absent": len(absent),
            "kind": kind,
        })

    examples = {}
    for name, rows in by_group.items():
        examples[name] = [{
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "answer": row["answer"],
            "jev": row["jev"],
            "completeness": row["completeness"],
            "gold_in_prompt": f"{row['gold_in_prompt']}/{row['n_gold']}",
            "facts_present": f"{row['facts_present']}/{row['n_facts']}",
            "refusal_phrase": row["refusal_phrase"],
        } for row in rows[:3]]

    report = {
        "wrong": {"n": len(wrong), "counts": counts,
                  "by_type": {t: dict(by_type[t]) for t in FOCUS if t in by_type}},
        "examples": examples,
        "flips": {
            "n": len(flips),
            "lost_supporting": sum(bool(row["lost_supporting"]) for row in flips),
            "same_documents": sum(row["same_documents"] for row in flips),
            "rows": flips,
        },
        "conflicting_info": conflicting,
        "info_not_found_flip": {
            **{k: v for k, v in guard_flip.items() if k != "kept"},
            "kept": [{**k, "passage_chars": len(k["passage"])} for k in guard_flip["kept"]],
        },
        "completeness": {
            "correct_but_incomplete": len(incomplete),
            "missing_facts_in_prompt": fact_in,
            "missing_facts_absent": fact_out,
            "answers": dict(answer_split),
        },
    }
    # The passage itself is large; keep it beside the summary.
    report["info_not_found_flip"]["passages"] = guard_flip["kept"]
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _print(report, guard_flip)
    return 0


def _print(report: dict, guard_flip: dict) -> None:
    print("\n=== 1. Wrong focus answers ===")
    print(report["wrong"]["counts"], "total", report["wrong"]["n"])
    print(f"{'type':28s}" + "".join(f"{name[:12]:>14s}" for name in GROUPS))
    for qtype, counts in report["wrong"]["by_type"].items():
        cells = "".join(f"{counts.get(name, 0):14d}" for name in GROUPS)
        print(f"{qtype:28s}{cells}")
    for name in GROUPS:
        print(f"\n--- {name} ---")
        for ex in report["examples"][name]:
            print(f"\n[{ex['question_type']}] {ex['question_id']} gold {ex['gold_in_prompt']} "
                  f"facts {ex['facts_present']} complete {ex['completeness']}")
            print(f"Jev: {ex['jev']}")
            if ex["refusal_phrase"]:
                print(f"refusal: {ex['refusal_phrase']}")
            print(f"Q: {_clip(ex['question'], 400)}")
            print(f"Gold: {_clip(ex['gold_answer'], 500)}")
            print(f"Ours: {_clip(ex['answer'], 500)}")

    flips = report["flips"]
    print("\n=== 2. Right to wrong ===")
    print(f"lost a gold or evidence doc: {flips['lost_supporting']} of {flips['n']}")
    print(f"same document set: {flips['same_documents']}")
    for row in flips["rows"]:
        if not row["lost_supporting"]:
            continue
        lost = ", ".join(f"{item['doc_id']} ({item['role']}: {item['title']})" for item in row["lost_supporting"])
        print(f"  {row['question_id']} {row['question_type']}: {lost}")

    print("\n=== 3. Conflicting info ===")
    print(f"{'id':12s}{'need':>6s}{'p22':>6s}{'fused':>7s}{'ok':>4s}")
    for row in report["conflicting_info"]:
        print(f"{row['question_id']:12s}{row['gold_needed']:6d}{row['in_phase22']:6d}"
              f"{row['in_fused']:7d}{str(row['correct']):>4s}")

    print("\n=== 4. info_not_found flip ===")
    print(guard_flip["question_id"])
    print("Q:", guard_flip["question"])
    print("Gold:", _clip(guard_flip["gold_answer"], 800))
    print("Phase 2.2 Jev:", guard_flip["phase22_jev"])
    print("Phase 2.2:", _clip(guard_flip["phase22_answer"], 800))
    print("Fused Jev:", guard_flip["fused_jev"])
    print("Fused:", _clip(guard_flip["fused_answer"], 800))
    for kept in guard_flip["kept"]:
        print(f"kept {kept['doc_id']} p={kept['jev_score']} {kept['title']}")
        print(_clip(kept["passage"], 1200))

    comp = report["completeness"]
    print("\n=== 5. Correct but incomplete ===")
    print(comp)


if __name__ == "__main__":
    raise SystemExit(main())
