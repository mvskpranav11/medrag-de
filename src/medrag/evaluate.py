"""Golden-set evaluation of the four retrieval modes.

Retrieval: Recall@5 and MRR@10 against the gold document pages.
Answers:   refusal rate (unanswerable), false-refusal rate (answerable), median latency,
           and RAGAS faithfulness + answer relevancy with a configurable judge LLM.
Only items with verified_by_human = true are used.

Usage: python -m medrag.evaluate [--modes dense bm25] [--limit 5] [--no-ragas] [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medrag.config import Settings, get_settings
from medrag.generate import Answer, answer_question, make_llm, select_context
from medrag.retrieval import MODES, Hit, Mode, Retrieval, Retriever

GOLDEN_PATH = Path("eval/golden_set.jsonl")
RESULTS_DIR = Path("eval/results")
README_PATH = Path("README.md")
MARKER_START, MARKER_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
TOP_K = 5  # sources given to the LLM; also the k of Recall@k
MRR_K = 10
JUDGE_TIMEOUT_S = 600.0  # a 7B judge on 8 GB RAM can be very slow
_RENDERED_CITATION = re.compile(r"\s*\[[^\[\]]+, p\. \d+\]")


@dataclass
class ItemResult:
    id: str
    question: str
    mode: str
    language: str
    answerable: bool
    cross_lingual: bool
    gold_rank: int | None  # 1-based rank of the first gold page in the top MRR_K, if any
    found: bool
    answer: str
    context_ids: list[str]  # chunk ids only, so no document text ends up in Git
    retrieval_ms: float
    generation_ms: float
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    judge_errors: list[str] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms


class FixedRetriever:
    """Hands answer_question a retrieval that was already done (and timed) once."""

    def __init__(self, retrieval: Retrieval) -> None:
        self.retrieval = retrieval

    def search(self, query: str, mode: str, k: int) -> Retrieval:
        return Retrieval(self.retrieval.hits[:k], self.retrieval.weak_evidence)


def load_golden(path: Path = GOLDEN_PATH) -> tuple[list[dict[str, Any]], int]:
    """Return the verified items and the total number of items."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r["verified_by_human"]], len(rows)


def gold_rank(hits: list[Hit], item: dict[str, Any]) -> int | None:
    pages = set(item["gold_pages"])
    for rank, hit in enumerate(hits, 1):
        if hit.doc_id == item["gold_doc_id"] and hit.page in pages:
            return rank
    return None


def run_item(
    item: dict[str, Any], mode: Mode, retriever: Retriever, llm: Any, settings: Settings
) -> ItemResult:
    start = time.perf_counter()
    retrieval = retriever.search(item["question"], mode, MRR_K)
    retrieval_ms = (time.perf_counter() - start) * 1000
    answer: Answer = answer_question(
        item["question"], mode, TOP_K, FixedRetriever(retrieval), llm, settings
    )
    contexts = select_context(retrieval.hits[:TOP_K], settings.max_context_tokens)
    return ItemResult(
        id=item["id"],
        question=item["question"],
        mode=mode,
        language=item["language"],
        answerable=item["answerable"],
        cross_lingual=item["cross_lingual"],
        gold_rank=gold_rank(retrieval.hits, item) if item["answerable"] else None,
        found=answer.found,
        answer=answer.answer,
        context_ids=[hit.chunk_id for hit in contexts],
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(answer.generation_ms, 1),
    )


def local_embeddings(embedder: Any) -> Any:
    """Lets RAGAS answer relevancy reuse the bge-m3 model that is already loaded."""
    from ragas.embeddings.base import BaseRagasEmbedding

    class LocalEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embedder.encode([text], normalize_embeddings=True)[0].tolist()

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.embed_text(text)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return embedder.encode(texts, normalize_embeddings=True).tolist()

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return self.embed_texts(texts)

    return LocalEmbeddings()


def judge_client(settings: Settings) -> Any:
    from openai import AsyncOpenAI

    if settings.judge_provider == "ollama":
        base_url, api_key = settings.ollama_url.rstrip("/") + "/v1", "ollama"
    else:
        base_url, api_key = settings.openai_base_url, settings.openai_api_key
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=JUDGE_TIMEOUT_S)


async def judge_all(results: list[ItemResult], settings: Settings, retriever: Retriever) -> None:
    """Score every answered item with RAGAS faithfulness and answer relevancy."""
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    llm = llm_factory(settings.judge_model, client=judge_client(settings), temperature=0)
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=local_embeddings(retriever.embedder))
    texts = retriever.chunk_texts()
    todo = [r for r in results if r.found and r.faithfulness is None and not r.judge_errors]
    for i, result in enumerate(todo, 1):
        response = _RENDERED_CITATION.sub("", result.answer).strip()
        try:
            score = await faithfulness.ascore(
                user_input=result.question,
                response=response,
                retrieved_contexts=[texts[i] for i in result.context_ids],
            )
            result.faithfulness = _finite(score.value)
        except Exception as error:  # a small judge sometimes returns invalid JSON
            result.judge_errors.append(f"faithfulness: {type(error).__name__}")
        try:
            score = await relevancy.ascore(user_input=result.question, response=response)
            result.answer_relevancy = _finite(score.value)
        except Exception as error:
            result.judge_errors.append(f"answer_relevancy: {type(error).__name__}")
        print(f"  judged {i}/{len(todo)} {result.mode} {result.id}", flush=True)


def _finite(value: Any) -> float | None:
    return None if value is None or math.isnan(float(value)) else round(float(value), 4)


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(results: list[ItemResult]) -> dict[str, Any]:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    faith = [r.faithfulness for r in results if r.faithfulness is not None]
    relevancy = [r.answer_relevancy for r in results if r.answer_relevancy is not None]
    return {
        "n_items": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "recall_at_5": _mean(
            [float(r.gold_rank is not None and r.gold_rank <= TOP_K) for r in answerable]
        ),
        "mrr_at_10": _mean([1 / r.gold_rank if r.gold_rank else 0.0 for r in answerable]),
        "faithfulness": _mean(faith),
        "answer_relevancy": _mean(relevancy),
        "n_judged": len(faith),
        "n_answered": sum(r.found for r in results),
        "refusal_rate_unanswerable": _mean([float(not r.found) for r in unanswerable]),
        "false_refusal_rate_answerable": _mean([float(not r.found) for r in answerable]),
        "median_latency_ms": round(statistics.median(r.total_ms for r in results))
        if results
        else None,
    }


def summarize_all(results: list[ItemResult], modes: list[str]) -> dict[str, Any]:
    subsets = {
        "all": lambda r: True,
        "de": lambda r: r.language == "de",
        "en": lambda r: r.language == "en",
        "cross_lingual": lambda r: r.cross_lingual,
    }
    return {
        mode: {
            name: summarize([r for r in results if r.mode == mode and keep(r)])
            for name, keep in subsets.items()
        }
        for mode in modes
    }


def environment(settings: Settings, n_verified: int, n_total: int, judged: bool) -> dict:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    llm_model = (
        settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model
    )
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "hardware": f"{sysctl('machdep.cpu.brand_string') or platform.processor()}, "
        f"{ram_gb:.0f} GB RAM, {platform.system()} {platform.machine()}",
        "llm": f"{settings.llm_provider}:{llm_model}",
        "judge": f"{settings.judge_provider}:{settings.judge_model}" if judged else "not run",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "query_device": settings.device,
        "top_k": TOP_K,
        "min_dense_score": settings.min_dense_score,
        "min_rerank_score": settings.min_rerank_score,
        "max_context_tokens": settings.max_context_tokens,
        "verified_items": n_verified,
        "total_items": n_total,
    }


def _fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "not measured"
    return f"{value * 100:.0f} %" if percent else f"{value:.2f}"


def markdown_table(summary: dict[str, Any], env: dict[str, Any]) -> str:
    lines = [
        "| Mode | Recall@5 | MRR@10 | Faithfulness | Answer relevancy | Refusal (unanswerable) "
        "| False refusal (answerable) | Median latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, subsets in summary.items():
        s = subsets["all"]
        latency = f"{s['median_latency_ms'] / 1000:.1f} s" if s["median_latency_ms"] else "–"
        lines.append(
            f"| `{mode}` | {_fmt(s['recall_at_5'])} | {_fmt(s['mrr_at_10'])} "
            f"| {_fmt(s['faithfulness'])} | {_fmt(s['answer_relevancy'])} "
            f"| {_fmt(s['refusal_rate_unanswerable'], True)} "
            f"| {_fmt(s['false_refusal_rate_answerable'], True)} | {latency} |"
        )
    lines += [
        "",
        "**By question language** (Recall@5 / false-refusal rate / faithfulness)",
        "",
        "| Mode | German questions | English questions |",
        "|---|---|---|",
    ]
    for mode, subsets in summary.items():
        cells = [
            f"{_fmt(subsets[lang]['recall_at_5'])} / "
            f"{_fmt(subsets[lang]['false_refusal_rate_answerable'], True)} / "
            f"{_fmt(subsets[lang]['faithfulness'])}"
            for lang in ("de", "en")
        ]
        lines.append(f"| `{mode}` | {cells[0]} | {cells[1]} |")
    all_counts = next(iter(summary.values()))
    de, en = all_counts["de"], all_counts["en"]
    lines += [
        "",
        f"Measured on {env['verified_items']} human-verified golden-set items "
        f"({all_counts['all']['n_answerable']} answerable, "
        f"{all_counts['all']['n_unanswerable']} unanswerable; "
        f"{de['n_items']} German, {en['n_items']} English questions), {env['date']}. "
        f"Hardware: {env['hardware']}. LLM: `{env['llm']}`, judge: `{env['judge']}`, "
        f"embeddings: `{env['embedding_model']}`, re-ranker: `{env['reranker_model']}`, "
        f"top-k {env['top_k']}. Faithfulness and answer relevancy are averaged over answered "
        "questions only. Details: [docs/EVALUATION.md](docs/EVALUATION.md).",
    ]
    return "\n".join(lines)


def update_readme(table: str, path: Path = README_PATH) -> None:
    text = path.read_text(encoding="utf-8")
    block = f"{MARKER_START}\n{table}\n{MARKER_END}"
    if MARKER_START in text and MARKER_END in text:
        before, rest = text.split(MARKER_START, 1)
        text = before + block + rest.split(MARKER_END, 1)[1]
    else:
        text = text.rstrip() + f"\n\n## Results\n\n{block}\n"
    path.write_text(text, encoding="utf-8")


def save_items(results: list[ItemResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_items(path: Path) -> list[ItemResult]:
    if not path.exists():
        return []
    return [ItemResult(**json.loads(line)) for line in path.read_text("utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--limit", type=int, help="only the first N verified items (smoke test)")
    parser.add_argument("--no-ragas", action="store_true", help="skip the LLM-judge metrics")
    parser.add_argument("--resume", action="store_true", help="reuse eval/results/items.jsonl")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    settings = get_settings()
    verified, n_total = load_golden()
    golden = verified[: args.limit] if args.limit else verified
    args.out.mkdir(parents=True, exist_ok=True)
    items_path = args.out / "items.jsonl"

    retriever = Retriever(settings)
    retriever.warm_up()
    llm = make_llm(settings)
    results = load_items(items_path) if args.resume else []
    done = {(r.id, r.mode) for r in results}
    for mode in args.modes:
        for item in golden:
            if (item["id"], mode) in done:
                continue
            results.append(run_item(item, mode, retriever, llm, settings))
            save_items(results, items_path)
            r = results[-1]
            print(
                f"{mode:14} {item['id']} found={r.found} rank={r.gold_rank} "
                f"{r.total_ms / 1000:.1f}s",
                flush=True,
            )

    if not args.no_ragas:
        asyncio.run(judge_all(results, settings, retriever))
        save_items(results, items_path)

    results = [r for r in results if r.mode in args.modes]
    judged = any(r.faithfulness is not None or r.answer_relevancy is not None for r in results)
    env = environment(settings, len(golden), n_total, judged)
    summary = summarize_all(results, args.modes)
    (args.out / "summary.json").write_text(
        json.dumps({"environment": env, "results": summary}, indent=2), encoding="utf-8"
    )
    table = markdown_table(summary, env)
    (args.out / "results.md").write_text(table + "\n", encoding="utf-8")
    if args.out == RESULTS_DIR and not args.limit:
        update_readme(table)
    print(table)


if __name__ == "__main__":
    main()
