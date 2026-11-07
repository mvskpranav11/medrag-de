# Evaluation

All numbers here come from one run of `make eval`. The raw answers per question are in
[`eval/results/items.jsonl`](../eval/results/items.jsonl), the aggregated numbers in
[`eval/results/summary.json`](../eval/results/summary.json).

> Demo project, not medical advice. These numbers describe retrieval and answer quality on a small
> test set, not clinical safety.

## Test set

[`eval/golden_set.jsonl`](../eval/golden_set.jsonl) has 40 questions about the 7 indexed documents.
Each has a reference answer, the document and page(s) where the answer is, the language, and
whether it is answerable. An LLM (Claude) drafted them. I then checked every question against the
PDFs and fixed answers and page numbers where needed. All 40 are marked `verified_by_human`, and
only verified questions are scored.

| | German | English | Total |
|---|---|---|---|
| Answerable, same language as the document | 11 | 11 | 22 |
| Answerable, other language than the document | 4 | 4 | 8 |
| Not answerable from the documents | 5 | 5 | 10 |

Page numbers are PDF pages (starting at 1), not the page numbers printed in the journals.

## What I compared

Four retrieval modes, with the same index, LLM, prompt and 5 sources per answer:

- `dense`: bge-m3 vectors in Qdrant
- `bm25`: keyword search
- `hybrid`: both, merged with Reciprocal Rank Fusion
- `hybrid_rerank`: hybrid, then the top 20 re-ranked by a cross-encoder

And these metrics:

- **Recall@5**: how often a correct page is in the top 5 chunks (answerable questions only)
- **MRR@10**: 1 / rank of the first correct page in the top 10, averaged (0 if not found)
- **Faithfulness** (RAGAS): share of the answer's statements that the retrieved text supports
- **Answer relevancy** (RAGAS): how well the answer fits the question
- **Refusal (unanswerable)**: how often it correctly says "not found". Higher is better.
- **False refusal (answerable)**: how often it says "not found" although the answer is there.
  Lower is better.
- **Median latency**: retrieval plus generation, per question

The two RAGAS metrics use an LLM as judge and only make sense for questions that were actually
answered. Citation markers are removed before judging.

Setup: Apple M3, 8 GB RAM, run natively (only Qdrant in Docker). LLM `qwen2.5:3b` via Ollama at
temperature 0, judge `qwen2.5:7b`, embeddings `BAAI/bge-m3` and re-ranker
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` on the CPU. "Not found" thresholds: dense score
0.60, re-ranker score 0.50. Context budget 3000 tokens.

## Results

| Mode | Recall@5 | MRR@10 | Faithfulness | Answer relevancy | Refusal (unanswerable) | False refusal (answerable) | Median latency |
|---|---|---|---|---|---|---|---|
| `dense` | 0.83 | 0.77 | 0.98 | 0.84 | 100 % | 7 % | 9.0 s |
| `bm25` | 0.67 | 0.54 | 1.00 | 0.79 | 90 % | 37 % | 7.5 s |
| `hybrid` | 0.90 | 0.68 | 0.88 | 0.82 | 100 % | 20 % | 7.8 s |
| `hybrid_rerank` | 0.97 | 0.83 | 0.95 | 0.84 | 90 % | 7 % | 8.8 s |

By question language (Recall@5 / false refusal / faithfulness):

| Mode | German questions | English questions |
|---|---|---|
| `dense` | 0.80 / 7 % / 0.96 | 0.87 / 7 % / 1.00 |
| `bm25` | 0.73 / 40 % / 1.00 | 0.60 / 33 % / 1.00 |
| `hybrid` | 0.87 / 20 % / 0.85 | 0.93 / 20 % / 0.92 |
| `hybrid_rerank` | 1.00 / 7 % / 0.96 | 0.93 / 7 % / 0.93 |

Only the 8 questions asked in the other language than the document:

| Mode | Recall@5 | MRR@10 | False refusal |
|---|---|---|---|
| `dense` | 7/8 | 0.80 | 0 % |
| `bm25` | 0/8 | 0.00 | 100 % |
| `hybrid` | 5/8 | 0.22 | 50 % |
| `hybrid_rerank` | 7/8 | 0.83 | 0 % |

Questions answered (the base for the two judge metrics): `dense` 28, `bm25` 20, `hybrid` 24,
`hybrid_rerank` 29 of 40.

What I take from this:

- Re-ranking helps the most. `hybrid_rerank` has the best Recall@5 and MRR and, together with
  `dense`, the fewest false refusals. For German questions it always had a correct page in the
  top 5.
- Merging BM25 into dense search without a re-ranker made the ranking worse (MRR 0.77 → 0.68),
  mostly because BM25 returns junk for cross-lingual questions.
- BM25 can't cross languages at all: 0 of 8. Most of its 37 % false refusals come from that.
- BM25's faithfulness of 1.00 doesn't mean much. It answered only 20 questions, mostly easy keyword
  matches, and its one wrong answer (case 3 below) still scored 1.0.
- `dense` and `hybrid` refused all 10 unanswerable questions. `bm25` and `hybrid_rerank` each
  answered one, both wrongly (cases 2 and 3). In `hybrid_rerank` the score gate stopped 9 of the
  10 before the LLM was even called.
- Most of the time is the LLM writing (median 7.5–10.6 s for answered questions). Retrieval takes
  about 0.01 s for BM25, 0.3 s for dense and hybrid, and 1.0 s with re-ranking.

## Five failure cases

All taken from `eval/results/items.jsonl`.

**1. BM25 and cross-lingual questions** (`bm25`, all 8 cross-lingual questions). For example
a13: "Welcher Zielblutdruck wird bei Hypertonie ohne Begleiterkrankungen empfohlen?" The answer
(< 140/90 mmHg) is on page 10 of the English WHO guideline. Words like *Zielblutdruck* simply
don't occur there, so BM25 found nothing and the answer was "not found". This is why the hybrid
modes need multilingual embeddings.

**2. An answer built from an off-topic passage** (`hybrid_rerank`, u03). "How many hours of sleep
should adults get per night?" No document covers this. But the re-ranker scored a passage from the
RKI sitting-time article above the threshold. The passage says that sleep should be left out
when counting sitting time. The 3B model turned that into "adults should get less than eight hours
of sleep per night", with a citation. The judge gave it 0.0. A single score threshold can't tell
"mentions sleep" from "answers the sleep question".

**3. Right numbers, wrong disease, and faithfulness didn't notice** (`bm25`, u08). "Wie lange
dauert die Inkubationszeit von Windpocken?" BM25 matched "Inkubationszeit" in a STIKO passage on
PDF page 59, which says the incubation period can be "zwischen < 10 Tagen und > 1 Jahr". That
passage is about **rabies**. The LLM presented it as the chickenpox incubation period, and cited
page 61 instead of 59. BM25 mode has no score gate. The judge gave faithfulness 1.0, because the
numbers really are in the retrieved text. Faithfulness checks "supported by the context", not
"correct".

**4. Right page found, still refused** (a11 and a19, all four modes). a19: "Is the 60 minutes of
activity per day for children a daily minimum?" Page 37 says the guideline changed it from a daily
minimum to a weekly average. a11 asks when the German national activity recommendations were first
published (2016/2017). The correct page was ranked first in every mode (a11 in `bm25`: third), and
the LLM still answered `NOT_FOUND`. All 21 false refusals in this run came from the LLM, none from
the score gate. My guess is that the 3B model is too cautious with yes/no questions and when the
wording of the question differs from the text. A bigger model or a few-shot example could help.
I haven't tested that.

**5. A fact about the wrong group of people** (`dense`, a16). "How often should adults do
muscle-strengthening activities?" Correct: 2 or more days a week (pages 12 and 42). Dense search
put the right page only at rank 9. One of the top 5 chunks said the 2010 WHO recommendations for
**children and adolescents** asked for "at least 3 days per week". The model answered "adults … at
least 3 days a week", and faithfulness was 1.0 again. `hybrid_rerank` found the right page and
answered correctly.

The 3B model's German is also sometimes garbled. In `hybrid` a15 it left citation markers where
the content should have been (faithfulness 0.0).

## Limitations

- 40 questions (30 answerable) is small. One question moves Recall@5 by 3.3 points, so small
  differences between modes don't mean much. It was one run, with no confidence intervals.
- The questions were drafted by an LLM. Even after checking, they may stay close to the wording of
  the source text.
- The judge (qwen2.5:7b) is small and from the same model family as the generator. I first tried
  qwen2.5:3b, but it marked a correct, cited answer as unsupported and called direct answers
  "noncommittal", which sets answer relevancy to 0. The 7B judge passed those checks, but cases 3
  and 5 show it measures support, not correctness. There is no correctness metric against the
  reference answers yet.
- I set the "not found" thresholds using 9 separate questions that are not in this test set, so
  they weren't tuned on the test data.
- Latency was measured natively on an 8 GB laptop that was swapping heavily. With `make up`,
  retrieval runs inside Docker's Linux VM and is slower (not measured).
- Recall counts a chunk as correct if it comes from a gold page, even if it's a different part of
  that page.

## Reproduce

```bash
make setup                 # dependencies, qwen2.5:3b and the qwen2.5:7b judge
docker compose up -d qdrant
make ingest
make eval                  # ~20 min for the answers, ~2 h for the 7B judge on 8 GB RAM
```

`python -m medrag.evaluate --no-ragas` skips the judge. `--resume` reuses saved answers, and
`--modes` runs only some modes (that run doesn't touch the README).
