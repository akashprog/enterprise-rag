# Beat the Leaderboard: Enterprise RAG

A 4-phase, build-it-yourself series showing why naive RAG fails on messy
company-internal data, and how to fix it step by step, measured on the
[Onyx EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
benchmark (~512k documents from a fictional company, "Redwood Inference", and 500 questions).

Development runs on **mini-redwood**, a 4,723-document subset: all 722 gold
documents plus 4,000 noise documents from Slack, Gmail, Linear and Confluence.
The final Phase 4 pipeline runs against the full corpus for leaderboard submission.

## Phases

| Phase | Folder | Idea |
|-------|--------|------|
| 0 | `scripts/`, `shared_utils/` | Curate mini-redwood, cost-aware API layer, local evaluator |
| 1 | `phase_1_baseline/` | Naive chunking + dense cosine search (the default trap) |
| 2 | `phase_2_data_mastery/` | Source-aware loaders, parent-child chunks, query expansion |
| 3 | `phase_3_jev_reranker/` | Hybrid dense + BM25 search, Jev yes/no (Noul) reranking |
| 4 | `phase_4_agent_router/` | LangGraph agent with a Jev Choice intent router |

## Results on mini-redwood

Metrics follow the benchmark paper: Overall = mean(correct × completeness).
Recall is Recall@10, and invalid extra docs is lower-is-better. Each phase is
listed with the cost of producing its answers. The corpus is the 5,389-document
mini-redwood, including 613 distractor documents, and the judge is the default
cascade. Judging a full run costs about $1.60.

| Phase | Overall | Correctness | Completeness | Recall@10 | Invalid extra docs | Cost |
|-------|---------|-------------|--------------|-----------|--------------------|------|
| 1 (fixed) | 46.31 | 51.40 | 59.98 | 75.06 | 6.41 | $0.34 answering (+$0.17 one-off ingest) |
| 2 | - | - | - | - | - | - |
| 3 | - | - | - | - | - | - |
| 4 | - | - | - | - | - | - |

"1 (fixed)": at `answer_max_tokens=1200`, 16 answers came back empty because hidden reasoning
used up the whole token limit. Only those 16 were re-answered with the limit raised to 4,000
(`--only-empty`); before the fix, Phase 1 scored 45.74. Phase 1 (fixed) is the baseline for Phase 2.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY (and TYPESAFE_API_KEY from Phase 3), fill MODEL_PRICES
docker compose up -d            # Qdrant on http://localhost:6333 (needed from Phase 1)
```

Download `all_documents.zip` from the
[latest Onyx release](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/latest)
and unzip it into `./all_documents/`.

## Step 1: curate and evaluate

```bash
# Build mini-redwood (free, ~30 s): writes data/raw_onyx_subset/*.jsonl
python scripts/curate_mini_redwood.py

# Score any phase's answers file ({question_id, answer, doc_ids} per line)
python -m shared_utils.evaluation results/phase_1/fixed/answers.jsonl --no-llm    # retrieval metrics only, $0
python -m shared_utils.evaluation results/phase_1/fixed/answers.jsonl --dry-run   # price the judge calls first
python -m shared_utils.evaluation results/phase_1/fixed/answers.jsonl             # full scoring (cascade judge)
python -m shared_utils.evaluation results/phase_1/fixed/answers.jsonl --judge sol # reference judge only, ~$4
```

### Judging: Jev first, uncertain verdicts to sol

Judging with `gpt-5.6-sol` alone cost ~$4 per 500 questions, more than 10x the cost of
answering them. The default `--judge cascade` asks Jev first, which costs ~$0.03 per
500 questions. Only verdicts that Jev is unsure about are re-asked to sol. The rules are
in `shared_utils/jev_judge.py`.

`scripts/compare_jev_judge.py` checked this against 485 cached sol verdicts on the
easy-corpus Phase 1 answers:

| Judge | Correctness agreement with sol | Fact agreement | Overall | Cost per 500 questions |
|---|---|---|---|---|
| sol only (reference) | 100% | 100% | 47.85 | ~$4.04 |
| Jev only | 92.6% (kappa 0.85) | 88.4% | - | ~$0.03 |
| **Cascade (default)** | **98.6%** | **98.1%** | **47.97** | **~$1.20** |

In cascade mode, about 28% of correctness verdicts and 30% of completeness verdicts
escalate to sol. Jev alone is too lenient on long answers that contain one small
conflicting detail, such as a wrong weekday or a version range. Those are exactly the
cases it marks as uncertain. The thresholds were tuned on these same questions, so
re-run the comparison when the corpus changes.

## Cost principles (applied everywhere)

- **Caching:** every paid call (embeddings, answers, judge verdicts, and later Jev)
  goes through `shared_utils/llm.py`, which caches results in `.cache/api_cache.sqlite`.
  Re-running unchanged work costs nothing.
- **Right-sized models:** small models answer, and Jev (a classifier) makes yes/no,
  judging and routing decisions. The strong model only judges the cases Jev is unsure of.
- **Cheap work first:** free local search (Qdrant, BM25) narrows candidates before
  any paid call.
- **Dry runs:** every tool supports `--dry-run` and `--limit` to price and trial a run
  before paying for it. Each run reports tokens and dollars per model.

## Layout

```text
all_documents/            unzipped Onyx corpus (gitignored)
data/raw_onyx_subset/     mini_redwood_docs.jsonl, mini_redwood_qa.jsonl (generated)
scripts/                  curate_mini_redwood.py, add_distractors.py, compare_jev_judge.py
shared_utils/             config, cache, llm (cost layer), evaluation (+ loaders in Phase 2)
phase_1_baseline/ ... phase_4_agent_router/
results/                  one folder per run; the index of every eval is results/README.md
run_eval.py               final submission entrypoint (Phase 4)
```
