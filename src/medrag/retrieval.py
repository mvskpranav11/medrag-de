"""Retrieval in four modes: dense, bm25, hybrid (RRF) and hybrid_rerank."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, get_args

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from medrag.config import Settings
from medrag.ingest import load_embedder, resolve_device

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
MODES: tuple[str, ...] = get_args(Mode)

RRF_K = 60  # standard constant from Cormack et al. (2009)
CANDIDATES = 30  # hits per retriever before fusion
RERANK_POOL = 20  # fused hits passed to the cross-encoder
RERANK_BATCH = 8


class Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


class Reranker(Protocol):
    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    language: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    weak_evidence: bool  # best calibrated score is below its threshold (or no hits)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_reranker(settings: Settings) -> Reranker:
    from sentence_transformers import CrossEncoder

    device = resolve_device(settings.device)
    model = CrossEncoder(settings.reranker_model, device=device, max_length=512)
    if device != "cpu":
        model.half()
    return model


def _hit(payload: dict[str, Any], score: float) -> Hit:
    return Hit(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        title=payload["title"],
        language=payload["language"],
        page=int(payload["page"]),
        text=payload["text"],
        score=float(score),
    )


class Retriever:
    """Holds the Qdrant client, an in-memory BM25 index and lazily loaded models."""

    def __init__(
        self,
        settings: Settings,
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        corpus: list[Hit] | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or QdrantClient(
            url=settings.qdrant_url, timeout=int(settings.request_timeout_s)
        )
        self._embedder = embedder
        self._reranker = reranker
        self._chunks: list[Hit] = []
        self._bm25: BM25Okapi | None = None
        if corpus is not None:
            self._index(corpus)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = load_embedder(self.settings, self.settings.device)
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = load_reranker(self.settings)
        return self._reranker

    def _index(self, chunks: list[Hit]) -> None:
        if not chunks:
            raise RuntimeError("No chunks indexed. Run `make ingest` first.")
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def _bm25_index(self) -> BM25Okapi:
        """Build the BM25 index from all chunks stored in Qdrant (once)."""
        if self._bm25 is None:
            chunks: list[Hit] = []
            offset = None
            while True:
                points, offset = self.client.scroll(
                    self.settings.qdrant_collection, limit=512, offset=offset, with_payload=True
                )
                chunks.extend(_hit(point.payload or {}, 0.0) for point in points)
                if offset is None:
                    break
            self._index(chunks)
        assert self._bm25 is not None
        return self._bm25

    def warm_up(self) -> int:
        """Load both models and the BM25 index up front. Returns the number of chunks."""
        self._bm25_index()
        self.embedder.encode(["warm-up"])
        self.reranker.predict([("warm-up", "warm-up")])
        return len(self._chunks)

    def count(self) -> int:
        return self.client.count(self.settings.qdrant_collection).count

    def chunk_texts(self) -> dict[str, str]:
        """All indexed chunk texts by chunk id."""
        self._bm25_index()
        return {hit.chunk_id: hit.text for hit in self._chunks}

    def dense(self, query: str, k: int) -> list[Hit]:
        vector = self.embedder.encode([query], normalize_embeddings=True)[0]
        response = self.client.query_points(
            self.settings.qdrant_collection,
            query=[float(v) for v in vector],
            limit=k,
            with_payload=True,
        )
        return [_hit(point.payload or {}, point.score) for point in response.points]

    def bm25(self, query: str, k: int) -> list[Hit]:
        scores = self._bm25_index().get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [replace(self._chunks[i], score=float(scores[i])) for i in top if scores[i] > 0]

    def hybrid(self, query: str, k: int) -> tuple[list[Hit], float]:
        """Fuse dense and BM25 rankings with RRF. Also returns the best dense cosine."""
        dense_hits = self.dense(query, CANDIDATES)
        sparse_hits = self.bm25(query, CANDIDATES)
        by_id = {hit.chunk_id: hit for hit in sparse_hits + dense_hits}
        fused = rrf([[h.chunk_id for h in dense_hits], [h.chunk_id for h in sparse_hits]])
        best_cosine = dense_hits[0].score if dense_hits else 0.0
        return [replace(by_id[cid], score=score) for cid, score in fused[:k]], best_cosine

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Re-score hits with the cross-encoder; scores are sigmoid probabilities (0..1)."""
        if not hits:
            return []
        logits = self.reranker.predict(
            [(query, hit.text) for hit in hits],
            batch_size=RERANK_BATCH,
            activation_fn=lambda x: x,
        )
        rescored = [
            replace(hit, score=sigmoid(float(s))) for hit, s in zip(hits, logits, strict=True)
        ]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)

    def search(self, query: str, mode: Mode, k: int) -> Retrieval:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        min_dense = self.settings.min_dense_score
        if mode == "dense":
            hits = self.dense(query, k)
            return Retrieval(hits, weak_evidence=not hits or hits[0].score < min_dense)
        if mode == "bm25":
            hits = self.bm25(query, k)
            return Retrieval(hits, weak_evidence=not hits)
        pool = RERANK_POOL if mode == "hybrid_rerank" else k
        hits, best_cosine = self.hybrid(query, max(pool, k))
        if mode == "hybrid":
            return Retrieval(hits, weak_evidence=not hits or best_cosine < min_dense)
        hits = self.rerank(query, hits)[:k]
        weak = not hits or hits[0].score < self.settings.min_rerank_score
        return Retrieval(hits, weak_evidence=weak)
