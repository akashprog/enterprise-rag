# Results index

One directory per run. This file says what each run was for, the config that
produced it, and the headline numbers. `log.jsonl` (created automatically) gets
one line appended every time an answering run or an evaluation finishes, so a
new run is never only on disk with no record. Add a section here when a run
becomes a baseline or changes a conclusion.

Scores are Overall / Correctness / Completeness / Recall@10 / Invalid extra docs.
Overall = mean(correct × completeness). Invalid extra docs is lower-is-better and
is an upper bound (every non-gold document counts). Answer words are counted with
citations stripped.

## Layout

```
results/
  phase_1/<run>/                  answers.jsonl, answer_cost.json
    <judge>/                      answers_eval.jsonl, answers_metrics.json
  phase_2/<setting>/              same, for an answered run
  phase_2/contexts/<setting>/     free context build: no answers, no judge
  phase_2/ab/                     side-by-side comparisons
  analysis/                       diagnostics that span runs
  judge/comparison/               the Jev-vs-sol experiment
```

`<judge>` is `sol`, `jev` or `cascade`, so judging the same answers twice keeps
both verdicts. A phase script writes its next run into this layout on its own.

## Phase 1

Shared config: `gpt-5.6-luna` answers, reasoning effort `low`, prompt
`p1-naive-v1`, retrieval is cosine top-10 over Qdrant `p1_naive` with a
6,000-token context budget. Judge reasoning effort is `low` throughout.

### `phase_1/easy/` — no distractors

The first full run, on the 4,723-document corpus before distractors were added.
It is the reference the judge comparison was measured against.

| Judge | Overall | Correct | Complete | Recall | Invalid | Answer cost | Judge cost |
|---|---|---|---|---|---|---|---|
| `sol/` `gpt-5.6-sol` | 47.85 | 52.40 | 61.08 | 77.37 | 6.43 | $0.33 | ~$4.04 |
| `cascade/` Jev, unsure cases to sol | 47.97 | 53.40 | 60.23 | 77.37 | 6.43 | same answers | $0 (verdicts already cached) |

`answer_max_tokens` was 1,200. The sol and cascade scores differ by 0.12, which
is the judge, not the corpus.

### `phase_1/before_fix/` — distractors, empty answers

Same pipeline on the 5,389-document corpus (723 gold, 4,000 noise, 613
distractors, 53 high-level evidence). Sixteen answers came back empty because
hidden reasoning used up the 1,200-token cap. Kept so the fix can be compared
against exactly the run it replaced.

Cascade judge: **45.74 / 50.80 / 58.74 / 75.06 / 6.41**. Answering $0.30, judging $1.61.

### `phase_1/fixed/` — the Phase 1 baseline

The same answers as `before_fix`, except the 16 empty ones were re-answered
after `answer_max_tokens` was raised to 4,000 (`--only-empty`; the other 484
came from the cache). This is the number Phase 2 is compared against.

| Judge | Overall | Correct | Complete | Recall | Invalid | Words | Cost |
|---|---|---|---|---|---|---|---|
| `cascade/` official | **46.31** | 51.40 | 59.98 | 75.06 | 6.41 | - | judging $1.21 |
| `jev/` rescore for the Phase 2 A/B | 43.60 | 50.00 | 58.86 | 75.06 | 6.41 | 151 | $0 (cached) |

Answering cost $0.34 ($0.30 plus $0.043 for the 16). The cascade judging also
had one crashed run whose cost was not recorded; the $1.21 is the successful
re-run. Jev scores the same answers lower than the cascade, so compare Jev rows
only with other Jev rows.

## Phase 2 — small-to-big context

The only change from Phase 1 is the context. Matching is the same `p1_naive`
collection and embeddings; the prompt is Phase 1's unchanged. Both answered runs
below use `gpt-5.6-luna`, reasoning `low`, `answer_max_tokens` 4000, and are
judged with **Jev only**. The cascade judge has not been run on either yet.

The first answering attempt hit the account's credit limit partway through
(129 and 127 answers saved). `--only-empty` finished both. The costs below
include both attempts.

### `phase_2/n5_w1_t2000_b8000/` — arm A, small-to-big

Search the top 20 chunks, keep the best 5 documents, send a document whole when
it is under 2,000 tokens and otherwise a window of the matched chunks ± 1
neighbour, and stop at 8,000 tokens of context. Average prompt: 5,448 tokens.

Jev: **55.37 / 61.60 / 67.10 / 69.63 / 4.05**, 143 words. Answering $0.71, judging $0.039.

### `phase_2/plain_k40_b5370/` — arm B, the equal-size control

Phase 1's exact recipe with only the size changed: the top chunks in rank
order, joined the same way, trimmed to a 5,370-token budget (up to 40 chunks).
That budget was chosen so the average prompt matches arm A's 5,448 tokens, so
the comparison isolates the context *method* from "more tokens". It spreads
each question over about 21 documents.

Jev: **51.61 / 57.80 / 66.80 / 77.74 / 19.97**, 159 words. Answering $0.72, judging $0.040.

### `phase_2/ab/n5_w1_t2000_b8000_vs_plain_k40_b5370.json` — A against B

On the 330 questions where Phase 1 already retrieved every gold document,
correctness is 80.0% for A and 69.4% for B (63.9% for Phase 1 fixed, all under
Jev). Between the arms, 58 questions are right only in A and 23 right only in
B. Per-type numbers are in the JSON.

## `phase_2/p07_f1_k30_b10000/` — Step 2.2, the relevance filter, answered

Same passages as arm A. The search is deeper (200 chunks) and Jev decides which
of the top 30 documents to send: keep a passage when its probability is at least
0.7, always keep at least one, and stop at 10,000 tokens. Prompt version
`p2-relevance-v1` (the prompt text is still Phase 1's). Judged with Jev only.

Jev: **66.76 / 73.40 / 77.90 / 82.21 / 2.97**, 151 words. Answering $0.67,
judging $0.039. The filter itself cost $1.07 (15,000 cached calls). Against arm
A (55.37), 82 questions went wrong to right and 23 went right to wrong.
`project_related` is the one type that did not improve (21.27 vs 22.16).
Per-type numbers and the four cache checks are in
`relevance_filter/extra_checks.json` and this run's `vs_arm_a.json`.

### `phase_2/relevance_filter/` — the selection sweep, before answering

Jev scored every candidate passage once: one yes/no, "does this passage help
answer the question?", $1.07 total (15,000 calls). The answered run above is
the p ≥ 0.7, floor 1 row.

| Selection | Recall@10 | Set recall | Invalid | Docs kept | Prompt tokens |
|---|---|---|---|---|---|
| Arm A, fixed 5 documents | 69.63 | 69.63 | 4.05 | 5.0 | 5,448 |
| Top 10 documents, no filter | 77.74 | 77.74 | 8.90 | 10 | - |
| Oracle: gold only, inside the top 30 | 86.84 | 86.84 | 0 | 1.2 | - |
| Jev p ≥ 0.7, keep at least 1 | **82.21** | 82.45 | **2.97** | 4.0 | 5,097 |

Set recall is the fraction of gold documents found anywhere in the kept set.
Recall@10 only counts gold among the first 10, so a longer list does not raise
it unless junk above the gold is removed. That is why the top-10, top-20, top-30
and top-50 lists all have Recall@10 of 77.74, while set recall climbs from 77.74
to 90.48. The oracle shows the headroom: dropping every non-gold document in the
top 30 would reach 86.84. Score cutoffs do not get there; the best of them
(keep scores within 90% of the best) scores Recall@10 73.03, below just keeping
the top 10.

80 of 470 questions with gold documents miss at least one of them even in the
top 30 (142 of 741 gold documents). Most missed documents are Confluence pages
(55), then Slack (21).

The full threshold grid is in `sweep.json`. The retrieval rows above were
measured before any answers; the answered comparison is the section above.

## `phase_2/hybrid_check/` — BM25 against the vector search (not answered)

Local Okapi BM25 over the same 48,448 chunks the vector index holds. No model
calls. Document rank is the best chunk, as in the cosine search. Fusion is
reciprocal rank fusion (k = 60) of each method's top 50 documents.

Gold documents found in the top 30, of the 741 a question asks for: vector
599 (80.8%), BM25 663 (89.5%), fused 673 (90.8%). Weighted per question, so it
matches the earlier set-recall figure, those are 86.8%, 91.3% and 94.7%.
Of the 142 gold documents the vector top 30 missed, BM25 finds 95 and the
fused list finds 80. Fusion drops 6 gold documents the vector list had.
`report.json` has the per-type and per-source tables. Nothing here was answered.

## `phase_2/hybrid_candidates/` — Step 2.3, fused top 30

Same passages, same filter (p ≥ 0.7, at least one document, 10k tokens), same
prompt as Phase 2.2. Paid runs will cover the focus set plus the 20
info_not_found questions, and will leave completeness and project_related for
the Phase 4 router.

Phase 2.2 on the focus set, from the saved Jev verdicts: **71.50 / 78.57 /
78.85 / 85.73 / 2.91**, 97 words, 420 questions. info_not_found stays 100 and
is not part of that headline. Later comparisons use 71.50.

Part 1 (`part1.json`) compares four candidate lists. On the focus set, gold
found is 88.9% for the vector top 30, 95.7% for the fused top 30, 95.2% for
the union of each method's top 20, and 97.1% for the union of each top 30.
Part 2 scored the fused top 30 for the 420 focus questions and the 20
info_not_found questions. One request holds as many passages as fit, so this
was 1,146 calls, $0.9298. Nothing was answered.

On the focus set, the Phase 2.2 rule (p ≥ 0.7, at least one document, 10k
tokens, search order) keeps Recall@10 94.02 and 1.45 invalid extra docs,
against 85.73 and 2.91 for the vector top 30. `part2.json` has the threshold sweep.

## `phase_2/fused30_p07_jev/` — fused top 30, answered

Same filter, with passages sorted by Jev score. Focus set plus the 20
info_not_found questions. Luna answers, Jev judge. Prompt version `p2-hybrid-v1`.

Focus set: **79.23 / 85.95 / 86.03 / 94.02 / 1.42**, 91 words, against Phase 2.2's
71.50 / 78.57 / 78.85 / 85.73 / 2.91. Flips versus Phase 2.2: 51 wrong to right,
20 right to wrong. Answering $0.38, judging $0.03, filter $0.93. Zero answer
failures. info_not_found, separate: 95.00, one question flipped from right to wrong.

## `phase_2/p2_answer_v2_dev/` and `p2_answer_v2_labels_dev/` — prompt v2 on DEV

Same retrieval, filter, and passages as `fused30_p07_jev`. Only the answer
prompt changed. DEV is 210 focus questions (seed 42, `dev_holdout.json`);
HOLDOUT was not answered. The 20 info_not_found questions were included and
are not part of the DEV headline. Jev judge only.

Not adopted. The answer prompt is Phase 2.3 plus the floor line. These two
directories stay on disk as the record of the rejected prompt.

DEV, against Phase 2.3's 77.49 / 83.81 / 84.86, 88.5 words:

| Arm | Overall | Correct | Complete | Words | Answer | Judge |
|---|---|---|---|---|---|---|
| v2, no labels | 71.93 | 81.43 | 79.84 | 46.2 | $0.18 | $0.014 |
| v2 plus title/source labels | 71.59 | 82.38 | 79.06 | 44.4 | $0.18 | $0.012 |

info_not_found went from 95 to 100 on both arms. Dry-run upper bound was $1.26
per arm; the 4,000-token output cap is what made that figure large.

## `phase_2/p2_floor_line_dev/` — floor sentence

Phase 2.3 prompt, plus the floor sentence on the 22 questions whose only
passage was kept by the floor (3 DEV, 19 info_not_found). The other answers
stay the Phase 2.3 ones. HOLDOUT was not answered.

Phase 2 final is this pipeline: fused top 30, the Jev filter (p ≥ 0.7, floor
1, Jev order, 10k), passages, and the Phase 1 prompt with the floor line.
Query expansion is not part of it.

DEV is unchanged at 77.49 / 83.81 / 84.86 (words 88.5 to 88.3). info_not_found
goes from 95 to 100: `qst_0492` flips wrong to right, and `qst_0490` was
already correct so it was not re-answered. Answering $0.011, judging $0.001.

## `phase_2/hybrid_candidates/query_expansion_dev.json` — retrieval only

DEV only. Three rewrites and one hypothetical passage per question, fused with
vector and BM25 the same way as Phase 2.3. No answers. The hypothetical
passage alone is the best list: gold found 213/219 against Phase 2.3's
209/219, and it drops no gold document Phase 2.3 had. Generation $0.08,
embeddings under $0.001. Scoring the new passages would be about $0.17.

## `phase_2/p2_expand_passage_dev/` — hypothetical passage, answered

Not the base pipeline. Kept on disk. The candidate list was the fused top 30
of the original question and a hypothetical passage, used for search only.
The filter and the prompt were Phase 2.3 plus the floor line. Jev judge.

DEV, against the baseline's 77.49 / 83.81 / 84.86: **80.24 / 86.67 / 85.52**,
90 words. Recall@10 93.41 to 94.39. Flips: 8 wrong to right, 2 right to wrong.
info_not_found stays 100. Jev scoring $0.45, answering $0.13, judging $0.008.

## `phase_2/p2_expand_passage_holdout/` — one HOLDOUT confirmation

Same expansion pipeline, 210 HOLDOUT questions, Jev only, one run. Not adopted.
The base pipeline stays Phase 2.3 plus the floor line.

Against Phase 2.3 on HOLDOUT (80.97 / 88.10 / 87.20, recall 94.63, invalid
1.48, 94 words): **78.25 / 85.24 / 84.15**, recall 92.20, invalid 1.55, 94
words. Flips: 5 wrong to right, 11 right to wrong.

Full focus set, DEV answers plus this run, 420 questions, against Phase 2.3
(79.23 / 85.95 / 86.03, recall 94.02, invalid 1.42, 91 words): **79.24 /
85.95 / 84.83**, recall 93.29, invalid 1.44, 92 words. Flips: 13 and 13.

Expansion $0.09, Jev scoring $0.40, answering $0.14, judging $0.007.

## `phase_2/p2_parked_baseline/` — parked questions, Phase 2 final

The 60 completeness and project_related questions, on the final pipeline
(fused top 30, Jev filter, Phase 1 prompt with the floor line). No expansion.
Cascade is the headline judge; Jev-only is the same answers.

Cascade **34.95 / 43.33 / 69.33**, recall 78.23, 530 words, after the one empty
answer (`qst_0440`) was re-answered at a 16,000-token cap. Jev-only 30.97 /
36.67 / 75.34. Gold documents in the fused top 30: 250/299. Filter scoring
$0.13, answering $0.16, cascade judging $1.18 plus $0.10 for the repaired
answer. The DEV/HOLDOUT split is `hybrid_candidates/parked_dev_holdout.json`
(seed 42, multi-part 24/24, count/list 6/6). Cascade on the halves: DEV 32.01,
HOLDOUT 37.90.

## `phase_2/contexts/` — free context trials

No model was called. Each folder holds the context that *would* be sent
(`contexts.jsonl`, `summary.json`), retrieval metrics on the documents sent
(`retrieval_metrics.json`, no judge), and a fact-presence check
(`recall1_all.csv`) over the 330 questions where Phase 1 had recall 1. "Facts"
is the share of gold facts found in the prompt by word overlap, for the
questions Phase 1 got wrong / right.

| Setting | What it varies | Facts, wrong / right | Recall | Invalid | Prompt tokens | Est. answer cost |
|---|---|---|---|---|---|---|
| `n5_w1_t2000_b6000` | N=5, ±1, 6k budget | 63% / 75% | 69.63 | 3.99 | 5,070 | $0.68 |
| `n5_w1_t2000_b8000` | arm A's setting | 66% / 76% | 69.63 | 4.05 | 5,448 | $0.72 |
| `n5_w2_t2000_b8000` | window ±2 | 69% / 77% | 69.63 | 4.04 | 5,945 | $0.77 |
| `n5_w1_t2000_b10000` | 10k budget | 66% / 76% | 69.63 | 4.05 | 5,483 | $0.72 |
| `n5_w2_t2000_b10000` | ±2 and 10k | 69% / 77% | 69.63 | 4.05 | 6,000 | $0.77 |
| `n8_w1_t2000_b10000` | N=8 | 78% / 84% | 74.71 | 6.83 | 8,286 | $1.00 |
| `n10_w1_t2000_b12000` | N=10 | 80% / 87% | 77.52 | 8.55 | 10,155 | $1.19 |
| `n100_w0_t0_b5450` | superseded control | 66% / 81% | 77.59 | 17.88 | 5,451 | $0.72 |
| `plain_k40_b5400`, `plain_k40_b5440` | budget calibration for arm B | - | - | - | 5,476 / 5,518 | - |

`n100_w0_t0_b5450` grouped chunks by document, so it is not a clean control;
`plain_k40_b5370` above replaced it. The two `plain_k40` folders only calibrated
the budget and were not scored. Phase 1's own prompt is 1,644 tokens and holds
33% / 61% of the facts on these 330 questions.

## `analysis/` — diagnostics

| File | What it is |
|---|---|
| `doc_lengths.json` | Token lengths of all 5,389 documents, overall and per source. Median 1,407, 77% are under 2,000 tokens, which is why the whole-document threshold is 2,000. From `measure_doc_lengths.py`. |
| `phase1_recall1_wrong.csv` | The 118 Phase 1 answers (before the token-cap fix) that were wrong even though every gold document was retrieved. |
| `phase1_recall1_all.csv` | The same check including the correct answers, before the fix. |
| `phase1_fixed_recall1_all.csv` | The same 330 questions after the fix: 115 wrong, 215 correct. Wrong answers had 33% of gold facts in the prompt, correct ones 61%. |

## `judge/comparison/` — why the cascade is the default judge

`scripts/compare_jev_judge.py` re-judged the `phase_1/easy` answers with Jev and
compared against the cached sol verdicts.

| | Correctness agreement with sol | Fact agreement | Cost per 500 |
|---|---|---|---|
| sol only | 100% | 100% | $4.04 |
| Jev only | 92.6% (kappa 0.85) | 88.4% | ~$0.03 |
| Cascade | 98.6% | 98.1% | ~$1.16 estimated |

The cascade sends a verdict to sol when Jev's confidence is low: about 28% of
correctness verdicts and 29% of completeness verdicts. `verdicts.jsonl` is every
comparison, `disagreements.jsonl` the ones they split on. The thresholds were
tuned on these same questions.
