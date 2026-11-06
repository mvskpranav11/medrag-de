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

Measured on 40 human-verified golden-set items (30 answerable, 10 unanswerable; 20 German, 20 English questions), 2026-11-06 21:34 UTC. Hardware: Apple M3, 8 GB RAM, Darwin arm64. LLM: `ollama:qwen2.5:3b`, judge: `ollama:qwen2.5:7b`, embeddings: `BAAI/bge-m3`, re-ranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, top-k 5. Faithfulness and answer relevancy are averaged over answered questions only. Details: [docs/EVALUATION.md](docs/EVALUATION.md).
