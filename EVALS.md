# Evaluation output reference

`python -m shared_utils.evaluation <answers.jsonl>` (or `--evaluate` on any phase's
query script) writes two files into `--output-dir` (the phase scripts use
`results/<phase>/<run>/<judge>/`):

- `<answers-stem>_eval.jsonl` has one row per question, described below.
- `<answers-stem>_metrics.json` has averages overall and by question type (including
  `avg_answer_words`), plus the count of verdicts decided by each judge and the judging
  cost (`judge_cost_usd`, with a per-model breakdown in `judge_cost`).

Every completed run also appends one line to `results/log.jsonl`. The index of all
runs, with what each one was for, is `results/README.md`.

## Per-question row

Example:

```json
{"question_id": "qst_0001", "question_type": "basic", "n_gold": 1, "n_retrieved": 3,
 "recall_at_k": 1.0, "invalid_extra_docs": 2, "correct": true, "completeness": 1.0,
 "score": 1.0, "correctness_rationale": "jev: refusal=answers(1.00) main_point=0.97 contradicts=0.04",
 "unsupported_facts": [], "correctness_judge": "jev", "completeness_judge": "jev",
 "answer_words": 42}
```

### Which question

| Field | Meaning |
|---|---|
| `question_id` | Benchmark question ID. Links the row to `mini_redwood_qa.jsonl` and to the answers file. |
| `question_type` | Benchmark category: one of basic, semantic, intra_document_reasoning, project_related, constrained, conflicting_info, completeness, miscellaneous, high_level, info_not_found. Metrics are also reported per type. |

### Retrieval metrics (computed in code, no API calls)

| Field | Meaning |
|---|---|
| `n_gold` | Number of gold documents needed to answer the question. It is 0 for `high_level` and `info_not_found`. |
| `n_retrieved` | Number of *distinct* documents the pipeline retrieved. Several chunks from the same document count once. |
| `recall_at_k` | Share of gold documents in the top k (10) retrieved documents. `null` for types without gold documents. |
| `invalid_extra_docs` | Number of retrieved documents that are not gold. Lower is better. The official harness doesn't count documents its judge panel marks as relevant. We don't run that panel, so ours counts every non-gold document and is an upper bound. |

### Answer quality (from the judge)

| Field | Meaning |
|---|---|
| `correct` | Whether the answer is broadly consistent with the gold answer: no conflicting facts, numbers, names, dates or versions. `info_not_found` answers are correct only if they say the information isn't available. `null` in a dry run. |
| `completeness` | Share (0–1) of the gold `answer_facts` that the answer states or clearly implies. |
| `score` | `correct × completeness`, so a wrong answer scores 0 however complete it is. The **Overall** score is the average of this field. |
| `answer_words` | Words in the answer as the judge saw it (citations stripped). Longer answers cost more to generate and to judge, and can raise completeness just by covering more ground, so compare it across phases. |

### Judge details

| Field | Meaning |
|---|---|
| `correctness_rationale` | Why the correctness verdict was given. From sol it is a one-sentence explanation. From Jev it shows Jev's raw signals (explained below). For an empty answer it is `"empty answer"`. |
| `unsupported_facts` | The gold facts the judge found missing, vaguer or contradicted in the answer. |
| `correctness_judge` | Who decided `correct`: `"jev"` (Jev was confident), `"sol"` (Jev was unsure, so `gpt-5.6-sol` decided) or `"none"` (no call needed, e.g. an empty answer). |
| `completeness_judge` | Who decided `completeness`, with the same values. It is also `"none"` when the question has no gold facts. |

## Reading a Jev rationale

`jev: refusal=answers(1.00) main_point=0.97 contradicts=0.04`

| Signal | Jev question type | Meaning |
|---|---|---|
| `refusal=answers(1.00)` | Choice | Jev chose "answers" rather than "declines", with confidence 1.00. |
| `main_point=0.97` | Yes/no | 97% probability that the answer states the gold answer's main conclusion. |
| `contradicts=0.04` | Yes/no | 4% probability that the answer conflicts with the gold answer. |

Jev's verdict is decided in code (`shared_utils/jev_judge.py`):

- `info_not_found` questions: correct only if the answer declines.
- All other questions: correct only if the answer answers, `main_point >= 0.7` and `contradicts < 0.3`.
- Each fact counts as supported if its yes/no probability is at least 0.5.

## When the cascade escalates to sol

The default `--judge cascade` keeps Jev's verdict unless Jev is uncertain:

| Verdict | Escalated to sol when |
|---|---|
| Correctness | The Choice confidence is below 0.8, `main_point` is in [0.6, 0.85), or `contradicts` is in [0.2, 0.5). The last two don't apply to `info_not_found`. |
| Completeness | Any fact's probability is in [0.3, 0.7). sol then re-judges every fact for that question, in one call per group of up to 12 facts. Facts a group's reply leaves out are re-asked one at a time. |

Each `"sol"` value in `correctness_judge` is one paid sol call; a `"sol"` in
`completeness_judge` is one call per 12 facts. These drive the cost of an evaluation.
Other modes are `--judge sol` (the reference, about $4 per 500 questions) and
`--judge jev` (about $0.03 per 500 questions). The comparison against sol is in
`scripts/compare_jev_judge.py` and `results/judge_comparison/`.

The judge's settings (`JUDGE_MODEL`, `JUDGE_REASONING_EFFORT`, `judge_max_tokens`) are
fixed for the whole series, so every phase is scored by the same judge and cached
verdicts stay valid.
