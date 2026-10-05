"""Can BM25 and vector search tell a basic question from a semantic one?

No model is called. The vector list is the saved Phase 2.3 search. BM25 is the
same local index Phase 2.3 fused with. The best threshold is chosen on focus
DEV and then applied to focus HOLDOUT.

Usage:
    python -m phase_2_data_mastery.search_signals
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase_2_data_mastery.answer_prompt_v2 import PHASE23_DIR, load_split  # noqa: E402
from phase_2_data_mastery.expand_answer import OUT_DIR as DEV_EXPAND  # noqa: E402
from phase_2_data_mastery.holdout_confirm import OUT_DIR as HOLD_EXPAND  # noqa: E402
from phase_2_data_mastery.hybrid_check import build_bm25, tokenize  # noqa: E402
from phase_2_data_mastery.query_small_to_big import DocIndex  # noqa: E402
from phase_2_data_mastery.relevance_filter import load_candidates  # noqa: E402
from shared_utils.config import settings, setup_logging  # noqa: E402
from shared_utils.evaluation import QuestionScore, aggregate  # noqa: E402
from dataclasses import fields

REPORT_PATH = settings.paths.results_dir / "phase_2" / "hybrid_candidates" / "search_signals.json"
_FIELDS = {f.name for f in fields(QuestionScore)}
STOP = set("""
a an the of to and or for in on at by with from as is are was were be been being
it this that these those we our you your they their i not no do does did if then
than what which who whom whose when where why how into over after before about
across per
""".split())


def _dist(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)

    def at(p: float) -> float:
        return ordered[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "n": n,
        "min": round(ordered[0], 4),
        "p25": round(at(0.25), 4),
        "median": round(at(0.50), 4),
        "mean": round(sum(ordered) / n, 4),
        "p75": round(at(0.75), 4),
        "max": round(ordered[-1], 4),
    }


def _signals() -> tuple[dict, dict, dict]:
    questions = [q for q in load_candidates() if q.question_type in ("basic", "semantic")]
    if len(questions) != 300:
        raise RuntimeError(f"Expected 300 basic and semantic questions, got {len(questions)}.")
    index_bm25, meta, _ = build_bm25()
    docs = DocIndex()
    labels = {}
    signals: dict[str, dict[str, float | None]] = {
        "bm25_top": {},
        "bm25_gap_1_5": {},
        "overlap_top10": {},
        "same_rank1": {},
        "word_share": {},
    }
    for q in questions:
        labels[q.question_id] = q.question_type
        ranked = index_bm25.ranked_documents(q.question, meta, limit=10)
        if not ranked:
            for name in signals:
                signals[name][q.question_id] = None
            continue
        signals["bm25_top"][q.question_id] = ranked[0]["score"]
        signals["bm25_gap_1_5"][q.question_id] = None if len(ranked) < 5 else ranked[0]["score"] - ranked[4]["score"]
        vector = [c.doc_id for c in q.candidates[:10]]
        bm25_ids = [row["doc_id"] for row in ranked[:10]]
        signals["overlap_top10"][q.question_id] = float(len(set(vector) & set(bm25_ids)))
        signals["same_rank1"][q.question_id] = float(vector[0] == ranked[0]["doc_id"]) if vector else None
        words = {tok for tok in tokenize(q.question) if tok not in STOP and len(tok) > 2}
        doc_words = set(tokenize(docs.get(ranked[0]["path"]).text))
        signals["word_share"][q.question_id] = None if not words else len(words & doc_words) / len(words)
    return labels, signals, {q.question_id: q.question_type for q in questions}


def _rule(values: dict[str, float | None], labels: dict[str, str], dev: list[str]) -> dict:
    """Threshold and direction with the best DEV accuracy. Ties flag fewer questions."""
    rows = [(values[qid], labels[qid] == "semantic") for qid in dev if values.get(qid) is not None]
    if not rows:
        raise RuntimeError("DEV has no values for a signal.")
    cuts = sorted({value for value, _ in rows})
    best = None
    for direction in (">=", "<="):
        for cut in cuts:
            correct = flags = 0
            for value, semantic in rows:
                flagged = value >= cut if direction == ">=" else value <= cut
                correct += flagged == semantic
                flags += flagged
            acc = correct / len(rows)
            rank = (acc, -flags, direction, cut)
            if best is None or rank > best[0]:
                best = (rank, {"direction": direction, "threshold": cut, "dev_accuracy": round(acc, 4),
                               "dev_flagged": flags, "dev_n": len(rows)})
    return best[1]


def _accuracy(values, labels, rule, ids) -> dict:
    rows = [qid for qid in ids if values.get(qid) is not None]
    correct = flags = 0
    flagged_ids = []
    for qid in rows:
        value = values[qid]
        flagged = value >= rule["threshold"] if rule["direction"] == ">=" else value <= rule["threshold"]
        correct += flagged == (labels[qid] == "semantic")
        if flagged:
            flags += 1
            flagged_ids.append(qid)
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "flagged": flags,
        "flagged_ids": flagged_ids,
    }


def _load_eval(path: Path) -> dict[str, QuestionScore]:
    return {
        row.question_id: row
        for row in (QuestionScore(**{k: raw[k] for k in _FIELDS if k in raw})
                    for raw in map(json.loads, path.open()))
    }


def _counterfactual(flagged: set[str], focus: list[str]) -> dict:
    phase23 = _load_eval(PHASE23_DIR / "jev" / "answers_eval.jsonl")
    expanded = _load_eval(DEV_EXPAND / "jev" / "answers_eval.jsonl")
    expanded.update(_load_eval(HOLD_EXPAND / "jev" / "answers_eval.jsonl"))
    missing = [qid for qid in flagged if qid not in expanded]
    if missing or set(focus) - set(phase23):
        raise RuntimeError(f"Expansion or Phase 2.3 is missing questions ({missing[:3]}).")
    mixed = []
    flips = {"wrong_to_right": 0, "right_to_wrong": 0}
    for qid in focus:
        base = phase23[qid]
        used = expanded[qid] if qid in flagged else base
        mixed.append(used)
        if qid in flagged:
            if not base.correct and used.correct:
                flips["wrong_to_right"] += 1
            elif base.correct and not used.correct:
                flips["right_to_wrong"] += 1
    packed = aggregate(mixed)
    base_packed = aggregate([phase23[qid] for qid in focus])
    return {
        "expansion_calls": len(flagged),
        "phase23": base_packed["overall"],
        "mixed": packed["overall"],
        "flips": flips,
    }


def main() -> int:
    setup_logging()
    labels, signals, _ = _signals()
    dev, hold = load_split()
    dev_bs = [qid for qid in dev if labels.get(qid) in ("basic", "semantic")]
    hold_bs = [qid for qid in hold if labels.get(qid) in ("basic", "semantic")]
    focus = [qid for qid in dev + hold if True]
    if len(focus) != 420:
        raise RuntimeError(f"Focus split is not 420 ({len(focus)}).")
    report = {"dev_n": len(dev_bs), "holdout_n": len(hold_bs), "signals": {}}
    best_name = None
    best_acc = -1.0
    for name, values in signals.items():
        basic = [values[qid] for qid, kind in labels.items() if kind == "basic" and values[qid] is not None]
        semantic = [values[qid] for qid, kind in labels.items() if kind == "semantic" and values[qid] is not None]
        rule = _rule(values, labels, dev_bs)
        held = _accuracy(values, labels, rule, hold_bs)
        report["signals"][name] = {
            "basic": _dist(basic),
            "semantic": _dist(semantic),
            "rule": {k: rule[k] for k in ("direction", "threshold", "dev_accuracy", "dev_flagged", "dev_n")},
            "holdout_accuracy": held["accuracy"],
            "holdout_flagged": held["flagged"],
            "holdout_n": held["n"],
        }
        if rule["dev_accuracy"] > best_acc:
            best_acc = rule["dev_accuracy"]
            best_name = name
            best_values = values
            best_rule = rule
    all_bs = dev_bs + hold_bs
    flagged = set(_accuracy(best_values, labels, best_rule, all_bs)["flagged_ids"])
    # Flag means "predicted semantic", the side expansion would run on.
    report["best_signal"] = best_name
    report["best_rule"] = report["signals"][best_name]["rule"]
    report["flagged"] = {
        "n": len(flagged),
        "basic": sum(labels[qid] == "basic" for qid in flagged),
        "semantic": sum(labels[qid] == "semantic" for qid in flagged),
        "dev": sum(qid in set(dev_bs) for qid in flagged),
        "holdout": sum(qid in set(hold_bs) for qid in flagged),
    }
    report["counterfactual"] = _counterfactual(flagged, focus)
    # Drop the large id list from the accuracy helper's holdout copy; flagged ids stay counted.
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "signals"}, indent=2))
    for name, block in report["signals"].items():
        mark = "  <-- best on DEV" if name == best_name else ""
        print(f"\n{name}{mark}")
        print(f"  basic    {_fmt(block['basic'])}")
        print(f"  semantic {_fmt(block['semantic'])}")
        rule = block["rule"]
        print(f"  DEV {rule['dev_accuracy']:.1%} if semantic when value {rule['direction']} {rule['threshold']}"
              f"  ({rule['dev_flagged']}/{rule['dev_n']} flagged)")
        print(f"  HOLDOUT accuracy {block['holdout_accuracy']:.1%}")
    mixed = report["counterfactual"]
    print(f"\nIf expansion ran only on the {report['flagged']['n']} flagged questions:")
    print(f"  focus {mixed['phase23']['overall_score']:.2f} -> {mixed['mixed']['overall_score']:.2f}")
    print(f"  flips {mixed['flips']['wrong_to_right']} wrong->right, {mixed['flips']['right_to_wrong']} right->wrong")
    print(f"  expansion calls {mixed['expansion_calls']}")
    return 0


def _fmt(dist: dict) -> str:
    return (f"n={dist['n']} min={dist['min']} p25={dist['p25']} med={dist['median']} "
            f"mean={dist['mean']} p75={dist['p75']} max={dist['max']}")


if __name__ == "__main__":
    raise SystemExit(main())
