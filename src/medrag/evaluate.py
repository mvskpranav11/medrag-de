"""Evaluate the four retrieval modes on the human-verified golden set.

Writes eval/results/ (items.jsonl, summary.json, results.md) and the results table in README.md.

    python -m medrag.evaluate               # full run
    python -m medrag.evaluate --resume      # reuse saved answers, judge what is missing
    python -m medrag.evaluate --no-ragas    # skip the LLM-judge metrics
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
from medrag.generate import answer_question, make_llm, select_context
from medrag.retrieval import MODES, Hit, Retrieval, Retriever

GOLDEN_PATH = Path("eval/golden_set.jsonl")
RESULTS_DIR = Path("eval/results")
README_PATH = Path("README.md")
MARKER_START, MARKER_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
TOP_K = 5  # sources given to the LLM, and the k in Recall@k
MRR_K = 10
JUDGE_TIMEOUT_S = 600.0  # a 7B judge on 8 GB RAM is slow
CITATION = re.compile(r"\s*\[[^\[\]]+, p\. \d+\]")


@dataclass
class ItemResult:
    id: str
    question: str
    mode: str
    language: str
    answerable: bool
    cross_lingual: bool
    gold_rank: int | None  # rank of the first gold page in the top 10, None if missing
    found: bool
    answer: str
    context_ids: list[str]  # chunk ids only, so no document text ends up in Git
    retrieval_ms: float
    generation_ms: float
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    judge_errors: list[str] = field(default_factory=list)


class FixedRetriever:
    """Gives answer_question the retrieval we already ran (and timed) for the metrics."""

    def __init__(self, retrieval: Retrieval) -> None:
        self.retrieval = retrieval

    def search(self, query: str, mode: str, k: int) -> Retrieval:
        return Retrieval(self.retrieval.hits[:k], self.retrieval.weak_evidence)


def gold_rank(hits: list[Hit], item: dict) -> int | None:
    for rank, hit in enumerate(hits, 1):
        if hit.doc_id == item["gold_doc_id"] and hit.page in item["gold_pages"]:
            return rank
    return None


def run_item(item: dict, mode: str, retriever: Retriever, llm: Any, settings: Settings):
    start = time.perf_counter()
    retrieval = retriever.search(item["question"], mode, MRR_K)
    retrieval_ms = (time.perf_counter() - start) * 1000
    answer = answer_question(
        item["question"], mode, TOP_K, FixedRetriever(retrieval), llm, settings
    )
    sources = select_context(retrieval.hits[:TOP_K], settings.max_context_tokens)
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
        context_ids=[hit.chunk_id for hit in sources],
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(answer.generation_ms, 1),
    )


async def judge(results: list[ItemResult], settings: Settings, retriever: Retriever) -> None:
    """RAGAS faithfulness and answer relevancy for every answered question."""
    # ragas is an optional dependency (the `eval` extra), so it is imported here.
    from openai import AsyncOpenAI
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    embedder = retriever.embedder

    class Bgem3(BaseRagasEmbedding):  # reuse the bge-m3 model that is already loaded
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.embed_texts([text])[0]

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return embedder.encode(texts, normalize_embeddings=True).tolist()

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.embed_text(text)

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return self.embed_texts(texts)

    if settings.judge_provider == "ollama":
        base_url, api_key = f"{settings.ollama_url.rstrip('/')}/v1", "ollama"
    else:
        base_url, api_key = settings.openai_base_url, settings.openai_api_key
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=JUDGE_TIMEOUT_S)
    llm = llm_factory(settings.judge_model, client=client, temperature=0)
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=Bgem3())
    texts = retriever.chunk_texts()

    todo = [r for r in results if r.found and r.faithfulness is None and not r.judge_errors]
    for n, r in enumerate(todo, 1):
        answer = CITATION.sub("", r.answer).strip()  # judge the text, not the citation markers
        contexts = [texts[chunk_id] for chunk_id in r.context_ids]
        try:
            score = await faithfulness.ascore(
                user_input=r.question, response=answer, retrieved_contexts=contexts
            )
            r.faithfulness = clean_score(score.value)
            score = await relevancy.ascore(user_input=r.question, response=answer)
            r.answer_relevancy = clean_score(score.value)
        except Exception as error:  # the judge sometimes returns invalid JSON
            r.judge_errors.append(type(error).__name__)
        print(f"  judged {n}/{len(todo)} {r.mode} {r.id}", flush=True)


def clean_score(value: Any) -> float | None:
    return None if value is None or math.isnan(value) else round(float(value), 4)


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(results: list[ItemResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    faithfulness = [r.faithfulness for r in results if r.faithfulness is not None]
    relevancy = [r.answer_relevancy for r in results if r.answer_relevancy is not None]
    return {
        "n_items": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_answered": sum(r.found for r in results),
        "recall_at_5": mean(
            [float(r.gold_rank is not None and r.gold_rank <= TOP_K) for r in answerable]
        ),
        "mrr_at_10": mean([1 / r.gold_rank if r.gold_rank else 0.0 for r in answerable]),
        "faithfulness": mean(faithfulness),
        "answer_relevancy": mean(relevancy),
        "refusal_rate_unanswerable": mean([float(not r.found) for r in unanswerable]),
        "false_refusal_rate_answerable": mean([float(not r.found) for r in answerable]),
        "median_latency_ms": round(
            statistics.median(r.retrieval_ms + r.generation_ms for r in results)
        )
        if results
        else None,
    }


def summarize_mode(results: list[ItemResult]) -> dict:
    return {
        "all": summarize(results),
        "de": summarize([r for r in results if r.language == "de"]),
        "en": summarize([r for r in results if r.language == "en"]),
        "cross_lingual": summarize([r for r in results if r.cross_lingual]),
    }


def hardware() -> str:
    cpu = platform.processor()
    if platform.system() == "Darwin":  # e.g. "Apple M3"
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    return f"{cpu}, {ram_gb:.0f} GB RAM, {platform.system()} {platform.machine()}"


def setup_info(settings: Settings, n_items: int, judged: bool) -> dict:
    llm = settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "hardware": hardware(),
        "llm": f"{settings.llm_provider}:{llm}",
        "judge": f"{settings.judge_provider}:{settings.judge_model}" if judged else "not run",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "top_k": TOP_K,
        "min_dense_score": settings.min_dense_score,
        "min_rerank_score": settings.min_rerank_score,
        "max_context_tokens": settings.max_context_tokens,
        "verified_items": n_items,
    }


def fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "not measured"
    return f"{value * 100:.0f} %" if percent else f"{value:.2f}"


def markdown_table(summary: dict, info: dict) -> str:
    lines = [
        "| Mode | Recall@5 | MRR@10 | Faithfulness | Answer relevancy | Refusal (unanswerable) "
        "| False refusal (answerable) | Median latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, s in summary.items():
        a = s["all"]
        lines.append(
            f"| `{mode}` | {fmt(a['recall_at_5'])} | {fmt(a['mrr_at_10'])} "
            f"| {fmt(a['faithfulness'])} | {fmt(a['answer_relevancy'])} "
            f"| {fmt(a['refusal_rate_unanswerable'], True)} "
            f"| {fmt(a['false_refusal_rate_answerable'], True)} "
            f"| {a['median_latency_ms'] / 1000:.1f} s |"
        )
    lines += [
        "",
        "**By question language** (Recall@5 / false-refusal rate / faithfulness)",
        "",
        "| Mode | German questions | English questions |",
        "|---|---|---|",
    ]
    for mode, s in summary.items():
        de, en = (
            f"{fmt(s[lang]['recall_at_5'])} / "
            f"{fmt(s[lang]['false_refusal_rate_answerable'], True)} / "
            f"{fmt(s[lang]['faithfulness'])}"
            for lang in ("de", "en")
        )
        lines.append(f"| `{mode}` | {de} | {en} |")
    counts = next(iter(summary.values()))
    lines += [
        "",
        f"Measured on {info['verified_items']} human-verified golden-set items "
        f"({counts['all']['n_answerable']} answerable, {counts['all']['n_unanswerable']} "
        f"unanswerable; {counts['de']['n_items']} German, {counts['en']['n_items']} English "
        f"questions), {info['date']}. Hardware: {info['hardware']}. LLM: `{info['llm']}`, "
        f"judge: `{info['judge']}`, embeddings: `{info['embedding_model']}`, re-ranker: "
        f"`{info['reranker_model']}`, top-k {info['top_k']}. Faithfulness and answer relevancy "
        "are averaged over answered questions only. Details: "
        "[docs/EVALUATION.md](docs/EVALUATION.md).",
    ]
    return "\n".join(lines)


def update_readme(table: str, path: Path = README_PATH) -> None:
    """Replace the results table between the two markers in the README."""
    text = path.read_text(encoding="utf-8")
    before, rest = text.split(MARKER_START, 1)
    after = rest.split(MARKER_END, 1)[1]
    path.write_text(f"{before}{MARKER_START}\n{table}\n{MARKER_END}{after}", encoding="utf-8")


def save(results: list[ItemResult], path: Path) -> None:
    lines = [json.dumps(asdict(r), ensure_ascii=False) for r in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load(path: Path) -> list[ItemResult]:
    if not path.exists():
        return []
    lines = path.read_text("utf-8").splitlines()
    return [ItemResult(**json.loads(line)) for line in lines if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--resume", action="store_true", help="reuse saved answers")
    parser.add_argument("--no-ragas", action="store_true", help="skip the LLM-judge metrics")
    args = parser.parse_args()

    settings = get_settings()
    rows = [json.loads(line) for line in GOLDEN_PATH.read_text("utf-8").splitlines()]
    golden = [row for row in rows if row["verified_by_human"]]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    items_path = RESULTS_DIR / "items.jsonl"

    retriever = Retriever(settings)
    retriever.warm_up()
    llm = make_llm(settings)
    results = load(items_path) if args.resume else []
    done = {(r.id, r.mode) for r in results}
    for mode in args.modes:
        for item in golden:
            if (item["id"], mode) in done:
                continue
            r = run_item(item, mode, retriever, llm, settings)
            results.append(r)
            save(results, items_path)  # save after every question, so --resume can continue
            seconds = (r.retrieval_ms + r.generation_ms) / 1000
            print(f"{mode:14} {r.id} found={r.found} rank={r.gold_rank} {seconds:.1f}s", flush=True)

    if not args.no_ragas:
        asyncio.run(judge(results, settings, retriever))
        save(results, items_path)

    results = [r for r in results if r.mode in args.modes]
    judged = any(r.faithfulness is not None for r in results)
    info = setup_info(settings, len(golden), judged)
    summary = {mode: summarize_mode([r for r in results if r.mode == mode]) for mode in args.modes}
    summary_json = json.dumps({"setup": info, "results": summary}, indent=2)
    (RESULTS_DIR / "summary.json").write_text(summary_json, encoding="utf-8")
    table = markdown_table(summary, info)
    (RESULTS_DIR / "results.md").write_text(table + "\n", encoding="utf-8")
    if set(args.modes) == set(MODES):  # only a full run goes into the README
        update_readme(table)
    print(table)


if __name__ == "__main__":
    main()
