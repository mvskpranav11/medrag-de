# Evaluation

All numbers on this page come from one run of `make eval` (`python -m medrag.evaluate`).
Raw per-question output is in [`eval/results/items.jsonl`](../eval/results/items.jsonl), and the aggregated metrics are in
[`eval/results/summary.json`](../eval/results/summary.json).

> Demo project, **not medical advice**. These metrics describe retrieval and answer quality on
> a small test set. They say nothing about clinical safety.

## Method

**Golden set** ([`eval/golden_set.jsonl`](../eval/golden_set.jsonl)): 40 questions written against
the 7 indexed documents. Each item has a question, a reference answer, the gold document and page(s),
the question language, an `answerable` flag and `verified_by_human`. Claude drafted the items. A
human then checked every item against the PDFs and corrected answers and pages where needed.
**All 40 items are human-verified**, and only verified items are scored.

| | German questions | English questions | Total |
|---|---|---|---|
| Answerable, same language as the document | 11 | 11 | 22 |
| Answerable, cross-lingual (question ≠ document language) | 4 | 4 | 8 |
| Unanswerable (topic not covered by the documents) | 5 | 5 | 10 |

Page numbers are PDF page indices (1-based), not the page numbers printed in the journal.

**Modes compared** (same index, same LLM, same prompt, top-k = 5 sources):

- `dense`: bge-m3 vector search in Qdrant.
- `bm25`: keyword search (rank_bm25).
- `hybrid`: dense + BM25 fused with Reciprocal Rank Fusion (k = 60).
- `hybrid_rerank`: hybrid, then the top 20 are re-ranked with a multilingual cross-encoder.

**Metrics**

| Metric | Definition |
|---|---|
| Recall@5 | Share of answerable questions where a gold page is among the top 5 retrieved chunks |
| MRR@10 | Mean of 1 / rank of the first gold page in the top 10 (0 if none) |
| Faithfulness | RAGAS: share of the answer's statements supported by the retrieved context (LLM judge) |
| Answer relevancy | RAGAS: how well questions regenerated from the answer match the original question (LLM judge + bge-m3) |
| Refusal (unanswerable) | Share of unanswerable questions answered with "not found" (higher is better) |
| False refusal (answerable) | Share of answerable questions answered with "not found" (lower is better) |
| Median latency | Median of retrieval + generation time per question, over all questions |

Faithfulness and answer relevancy are only computed for questions the system actually answered.
Refusals have no statements to judge. Citation markers like `[title, p. 3]` are removed before
judging. Retrieval runs once per question and mode: the top 10 hits are used for MRR, and the top 5
of the same list go to the LLM.

**Setup:** Apple M3, 8 GB RAM, Darwin arm64. Native run (not Docker), Docker API/UI stopped, Qdrant in Docker.
LLM `ollama:qwen2.5:3b` (temperature 0), judge `ollama:qwen2.5:7b`, embeddings `BAAI/bge-m3` (CPU),
re-ranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (CPU), evidence thresholds dense ≥ 0.6,
re-rank ≥ 0.5, context budget 3000 tokens. Run date 2026-11-06 21:34 UTC.

## Results

| Mode | Recall@5 | MRR@10 | Faithfulness | Answer relevancy | Refusal (unanswerable) | False refusal (answerable) | Median latency |
|---|---|---|---|---|---|---|---|
| `dense` | 0.83 | 0.77 | 0.98 | 0.84 | 100 % | 7 % | 9.0 s |
| `bm25` | 0.67 | 0.54 | 1.00 | 0.79 | 90 % | 37 % | 7.5 s |
| `hybrid` | 0.90 | 0.68 | 0.88 | 0.82 | 100 % | 20 % | 7.8 s |
| `hybrid_rerank` | 0.97 | 0.83 | 0.95 | 0.84 | 90 % | 7 % | 8.8 s |

**By question language** (Recall@5 / false-refusal rate / faithfulness)

| Mode | German questions | English questions |
|---|---|---|
| `dense` | 0.80 / 7 % / 0.96 | 0.87 / 7 % / 1.00 |
| `bm25` | 0.73 / 40 % / 1.00 | 0.60 / 33 % / 1.00 |
| `hybrid` | 0.87 / 20 % / 0.85 | 0.93 / 20 % / 0.92 |
| `hybrid_rerank` | 1.00 / 7 % / 0.96 | 0.93 / 7 % / 0.93 |

Measured on 40 human-verified golden-set items (30 answerable, 10 unanswerable; 20 German, 20 English questions), 2026-11-06 21:34 UTC. Hardware: Apple M3, 8 GB RAM, Darwin arm64. LLM: `ollama:qwen2.5:3b`, judge: `ollama:qwen2.5:7b`, embeddings: `BAAI/bge-m3`, re-ranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, top-k 5. Faithfulness and answer relevancy are averaged over answered questions only. Details: this page.

**Cross-lingual questions only** (8 items, from `summary.json`):

| Mode | Recall@5 | MRR@10 | False refusal |
|---|---|---|---|
| `dense` | 7/8 | 0.80 | 0 % |
| `bm25` | 0/8 | 0.00 | 100 % |
| `hybrid` | 5/8 | 0.22 | 50 % |
| `hybrid_rerank` | 7/8 | 0.83 | 0 % |

**Answered questions per mode** (the base for the two judge metrics): `dense` 28,
`bm25` 20, `hybrid` 24, `hybrid_rerank` 29 out of 40.

### Reading the numbers

- **Re-ranking gives the best retrieval.** `hybrid_rerank` has the best Recall@5 (0.97) and MRR@10
  (0.83) and the lowest false-refusal rate (7 %, tied with `dense`). On German questions it found
  the gold page in the top 5 every time.
- **Plain RRF fusion is worse than dense alone on ranking** (MRR 0.68 vs 0.77). BM25 adds noise,
  especially for cross-lingual questions where it finds nothing useful. The cross-encoder fixes the
  order again. Without it, `hybrid` refuses 20 % of answerable questions.
- **BM25 cannot cross languages.** It retrieved no gold page for any of the 8 cross-lingual
  questions, so all 8 were refused. That accounts for most of its 37 % false-refusal rate.
- **BM25's perfect faithfulness (1.00) is not a sign of quality.** It answered only 20 of 40
  questions, mostly easy keyword matches, and its one wrong answer (failure case 3) still scored as
  faithful.
- **Refusing unanswerable questions works well.** `dense` and `hybrid` refused all 10. `bm25` and
  `hybrid_rerank` each answered one of them, and both answers were wrong (failure cases 2 and 3). In
  `hybrid_rerank`, the evidence gate stopped 9 of the 10 before the LLM was called.
- **Latency** is 7.5–9 s median on an 8 GB M3. Almost all of it is the 3B LLM generating text
  (median 7.5–10.6 s for answered questions). Median retrieval takes 0.01 s (BM25), 0.3 s (dense,
  hybrid) and 1.0 s (with re-ranking). Refusals by the gate return after retrieval alone.

## Five failure cases

Mode, item id and output are taken from `eval/results/items.jsonl`.

**1. BM25 cannot match across languages (`bm25`, all 8 cross-lingual items).**
Example a13: "Welcher Zielblutdruck wird bei Hypertonie ohne Begleiterkrankungen empfohlen?" The
answer is on page 10 of the English WHO guideline ("target … <140/90 mmHg"), but German words like
*Zielblutdruck* or *Begleiterkrankungen* never occur in the English text. BM25 returned no gold page
and the LLM answered "not found". *Likely cause:* lexical retrieval has no shared vocabulary between
the languages. This is the main reason to use multilingual dense embeddings (bge-m3), and why
the hybrid modes need dense retrieval.

**2. A plausible-sounding inference from an off-topic source (`hybrid_rerank`, u03).**
"How many hours of sleep should adults get per night?" No document covers this, but the re-ranker
scored a passage from the RKI sitting-time article above the 0.50 gate. The passage says that time spent sleeping is *excluded*
from sitting time. The 3B LLM turned this into "adults should get less than eight hours of
sleep per night", with a citation. The judge gave it faithfulness 0.0. *Likely cause:* a small
generator over-reaches when the context is only loosely related, and the re-ranker gate (one score
threshold) cannot tell "about sleep" from "answers the sleep question".

**3. Right numbers, wrong disease (`bm25`, u08). The faithfulness metric did not catch this one.**
"Wie lange dauert die Inkubationszeit von Windpocken?" BM25 retrieved a STIKO passage (PDF page 59)
saying the incubation period "zwischen < 10 Tagen und > 1 Jahr betragen kann". That sentence is
about **rabies** post-exposure prophylaxis. The LLM presented it as the incubation period of chickenpox,
and it cited page 61 instead of page 59. BM25 mode
has no evidence gate, so the LLM saw this context. The judge gave **faithfulness 1.0**, because the
statement's numbers are literally in the context. *Likely cause:* keyword match on "Inkubationszeit"
without topic check, plus a metric that checks support in the context, not correctness. This is why
faithfulness alone is not enough, and why the golden set also tracks refusals on unanswerable items.

**4. Correct page retrieved near the top, still refused (a11 and a19, all four modes).**
a19: "Is the 60 minutes of activity per day for children a daily minimum?" Page 37 says the
guideline changed the daily minimum to a weekly average. a11 asks when German national physical
activity recommendations were first published (2016/2017). The gold page ranked first in every mode (a11 in
`bm25`: third), yet the LLM answered `NOT_FOUND`. All 21 false refusals in this run came from the LLM, not from the
evidence gate. *Likely cause:* the 3B model is conservative on yes/no questions with a negation and
on questions where the answer is phrased differently from the question. A larger generator or a
few-shot example in the prompt are the obvious next tests (not measured).

**5. A fact from the wrong population (`dense`, a16).**
"How often should adults do muscle-strengthening activities?" The right answer is 2 or more days a
week (page 12/42). Dense retrieval ranked the gold page only 9th. One of the top-5 passages said that
the 2010 WHO recommendations for **children and adolescents** asked for muscle- and bone-strengthening
"at least 3 days per week". The LLM answered "adults … at least 3 days a week". Faithfulness: 1.0.
*Likely cause:* the retrieved chunk matched the topic but not the population, and the small LLM did
not check who the sentence was about. Hybrid + re-ranking retrieved the gold page and answered this
one correctly.

Also seen: German answers from the 3B model are sometimes garbled (e.g. `hybrid` a15 left
citation markers where the content should be, faithfulness 0.0).

## Limitations

- **Small test set.** 40 questions (30 answerable) from 7 documents. One question changes Recall@5 by
  3.3 points, so differences of a few points between modes are not meaningful. There are no confidence
  intervals and only one run.
- **The questions were drafted by an LLM, then human-verified.** They may still favour wording that
  is close to the source text.
- **The judge is local and small** (qwen2.5:7b, same model family as the generator). The first choice,
  qwen2.5:3b, was rejected after a manual check. It marked a correct, cited answer as unsupported and
  called direct answers "noncommittal", which forces answer relevancy to 0. The 7B judge got those
  checks right, but failure cases 3 and 5 show it still scores "supported by the context" rather than
  "correct". There is no answer-correctness metric against the reference answers.
- **Evidence thresholds** (dense 0.60, re-rank 0.50) were calibrated on 9 separate ad-hoc
  questions that are **not** in the golden set, to avoid tuning on the test data.
- **Latency** was measured on an 8 GB Apple M3 with heavy memory swapping, natively (not in Docker).
  In Docker, without GPU access for Ollama, it is higher.
- **Recall is page-level.** A chunk counts as a hit if it is from a gold page, even if it is a
  different part of that page.

## Reproduce

```bash
make setup          # installs dependencies, pulls qwen2.5:3b and the qwen2.5:7b judge
docker compose up -d qdrant
make ingest
make eval           # about 20 min for answers + about 2 h for the 7B judge on 8 GB RAM
```

`python -m medrag.evaluate --no-ragas` skips the judge metrics. `--resume` reuses saved answers,
and `--modes`/`--limit` run a subset (a subset run does not update the README).
