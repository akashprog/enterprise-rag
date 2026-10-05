# Phase 1: The Default Trap (naive RAG baseline)

The textbook LangChain recipe, unchanged, on messy enterprise data:

1. **Ingest** (`ingest_naive.py`): each document becomes one flat string (title line + body).
   `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` splits it, each chunk is
   embedded with `text-embedding-3-small`, and the chunks go into Qdrant collection `p1_naive`
   (cosine distance).
2. **Query** (`query_naive.py`): embed the question, take the top 10 chunks by cosine similarity,
   paste them into a plain "answer from the context" prompt, and make one `gpt-5.6-luna` call.
   The documents behind those 10 chunks are reported as retrieved, with nothing filtered out.

## Run it

```bash
docker compose up -d
python -m phase_1_baseline.ingest_naive --dry-run          # 48,448 chunks, ~8.7M tokens, ~$0.17
python -m phase_1_baseline.ingest_naive                    # ~8 min; re-runs are free
python -m phase_1_baseline.query_naive --dry-run           # price the answering run
python -m phase_1_baseline.query_naive --limit 20 --evaluate
python -m phase_1_baseline.query_naive --evaluate          # all 500 questions + scoring
```

Outputs: `results/phase_1/fixed/` (`answers.jsonl`, `answer_cost.json`, and
`<judge>/answers_eval.jsonl` plus `answers_metrics.json`). Every run is indexed in
`results/README.md`.

## Results (mini-redwood, 500 questions)

This is the baseline that later phases are compared against. The corpus has 5,389
documents:
- 723 gold documents;
- 4,000 random noise documents;
- 613 distractors: hard negatives that share BM25 keywords with a specific question
  (`scripts/add_distractors.py`);
- 53 evidence documents, which make the `high_level` questions answerable.

The judge is the default cascade: Jev first, with uncertain verdicts sent to `gpt-5.6-sol`.
Overall = mean(correct × completeness). Recall is Recall@10, and invalid extra docs is
lower-is-better.

| Question type | n | Overall | Correct | Complete | Recall@10 | Invalid extra | Overall without distractors |
|---|---|---|---|---|---|---|---|
| **OVERALL** | 500 | **45.74** | 50.80 | 58.74 | 75.06 | 6.41 | 47.85 |
| basic | 175 | 56.37 | 61.14 | 62.18 | 81.71 | 6.78 | 58.84 |
| semantic | 125 | 32.13 | 36.80 | 45.64 | 62.40 | 7.50 | 34.86 |
| intra_document_reasoning | 40 | 43.75 | 45.00 | 63.66 | 85.00 | 6.47 | 45.42 |
| project_related | 40 | 16.10 | 25.00 | 50.23 | 62.77 | 3.48 | 17.91 |
| constrained | 30 | 50.10 | 53.33 | 80.14 | 93.33 | 4.70 | 52.22 |
| conflicting_info | 20 | 45.36 | 55.00 | 58.48 | 87.50 | 5.40 | 57.64 |
| completeness | 20 | 22.92 | 30.00 | 30.85 | 40.85 | 6.35 | 28.33 |
| miscellaneous | 20 | 68.00 | 80.00 | 73.00 | 95.00 | 5.65 | 55.83 |
| high_level | 10 | 46.67 | 50.00 | 67.67 | - | - | 43.33 |
| info_not_found | 20 | 95.00 | 95.00 | 95.00 | - | - | 95.00 |

The last column is the earlier run on a 4,723-document corpus with no distractors, judged by
sol alone (archived in `results/phase_1/easy/sol/`). The cascade judge scores those same easy answers
47.97, so the judge accounts for about 0.1 points of the gap and the distractors for the rest.

**Distractors at work:**
- 480 questions have their own distractor documents. For 56% of them (270 questions), at
  least one of those distractors made it into the top 10.
- Distractors make up 21% of all retrieved documents.
- Questions that retrieved their own distractor lost 3.4 Overall points on average, against
  0.6 points for questions that didn't.
- Conflicting Info lost the most (-12.3), because its distractors are the outdated versions
  of the answer.
- Miscellaneous went up, because recall rose there (80 → 95). Qdrant's approximate HNSW
  index was rebuilt during re-ingest, so small shifts in rankings are expected.

**Cost and time:**
- One-off ingest: $0.17 (8.7M embedding tokens; re-ingesting only embedded the new documents).
- Answering all 500 questions: **$0.30** (822k input / 112k output tokens on `gpt-5.6-luna`;
  70 answers came from the cache), 194 s at concurrency 8.
- Judging: **$1.61** in 122 s.
  - Jev: 810 calls, $0.03.
  - sol: 232 escalated calls, $1.58. Escalated cases are the hard ones, so sol produced
    about 190 output tokens per call here, against about 100 when it judged everything.
  - Judging the easy run with sol alone cost about $4.04 and took 189 s.

## What the numbers say

- **Retrieval is noisy.** Each question returns 7.4 documents on average, of which about 6.4
  are not gold for that question. Those extra documents are the "Invalid Extra Docs" the
  leaderboard penalizes, and they compete for the model's attention.
- **Finding the gold document isn't enough.** When every gold document was retrieved
  (recall = 1, 330 questions), the answer was still correct only **64.2%** of the time.
  The relevant chunk is there, but it's surrounded by near-duplicates, distractors and
  similar-sounding documents from other projects. When no gold document was retrieved
  (91 questions), correctness drops to 7.7%.
- **Multi-document questions break first.** Project Related (16.1) and Completeness (22.9)
  need facts spread over 2 to 10 documents. With fixed 10-chunk retrieval, several chunks
  often come from the same document, so most of the gold set never reaches the prompt
  (Completeness recall is only 41%).
- **Semantic questions show the limits of embeddings.** Questions phrased with little
  keyword overlap score 32.1 overall, with 62% recall, against 81.7% recall for Basic
  questions. Dense similarity on 1,000-character fragments misses roundabout phrasing.
- **Info Not Found looks fine (95), for the wrong reason.** The tutorial prompt's "say you
  don't know" line works when nothing relevant is retrieved. The same habit hurts elsewhere,
  when relevant evidence is buried among noise.

For context, the paper's full-corpus vector baseline (text-embedding-3-large) reports 51.4%
correctness, 46.0% recall and 9.3 invalid extra docs. Our recall is much higher because
mini-redwood is about 100 times smaller than the full corpus, so there are far fewer
distractors. Compare phases with each other, not with the leaderboard, until Phase 4 runs
on the full corpus.

## What Phase 2 targets

- Thread- and email-aware loaders, so a chunk keeps its conversation context.
- Parent-child chunks: match small chunks, hand the model the whole section, and count each
  document once.
- Query expansion for codenames and acronyms, which targets the semantic and completeness gaps.
