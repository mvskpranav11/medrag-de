from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from medrag.config import Settings
from medrag.retrieval import Hit, Retriever, rrf, sigmoid, tokenize

CORPUS = [
    Hit("c1", "who-pa", "WHO PA", "en", 3, "adults physical activity 150 minutes per week", 0.0),
    Hit("c2", "who-htn", "WHO HTN", "en", 7, "hypertension treatment blood pressure 140 mmHg", 0.0),
    Hit("c3", "stiko", "STIKO", "de", 12, "Impfung gegen Masern für Kinder", 0.0),
]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.calls += 1
        return np.ones((len(sentences), 4), dtype="float32")


class FakeClient:
    """Returns the corpus ranked by fixed cosine scores, like Qdrant's query_points."""

    def __init__(self, cosines: dict[str, float]) -> None:
        self.cosines = cosines
        self.calls = 0

    def query_points(self, collection: str, query: list[float], limit: int, **kwargs: object):
        self.calls += 1
        ranked = sorted(CORPUS, key=lambda h: self.cosines[h.chunk_id], reverse=True)[:limit]
        points = [
            SimpleNamespace(payload=asdict(h), score=self.cosines[h.chunk_id]) for h in ranked
        ]
        return SimpleNamespace(points=points)


class FakeReranker:
    def __init__(self, logits: dict[str, float]) -> None:
        self.logits = logits

    def predict(self, inputs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        by_text = {h.text: self.logits[h.chunk_id] for h in CORPUS}
        return [by_text[text] for _, text in inputs]


def make_retriever(cosines: dict[str, float], logits: dict[str, float] | None = None):
    embedder, client = FakeEmbedder(), FakeClient(cosines)
    retriever = Retriever(
        Settings(_env_file=None),
        client=client,
        embedder=embedder,
        reranker=FakeReranker(logits or {"c1": 0.0, "c2": 0.0, "c3": 0.0}),
        corpus=CORPUS,
    )
    return retriever, embedder, client


def test_rrf_maths() -> None:
    fused = dict(rrf([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)
    assert [item for item, _ in rrf([["a", "b"], ["b", "c"]])] == ["b", "a", "c"]


def test_tokenize_keeps_umlauts() -> None:
    assert tokenize("Bluthochdruck bei Älteren?") == ["bluthochdruck", "bei", "älteren"]


def test_bm25_mode_uses_no_embeddings() -> None:
    retriever, embedder, client = make_retriever({"c1": 0.9, "c2": 0.1, "c3": 0.1})
    result = retriever.search("hypertension treatment", "bm25", k=2)
    assert result.hits[0].chunk_id == "c2"
    assert embedder.calls == 0 and client.calls == 0


def test_dense_mode_flags_weak_evidence() -> None:
    retriever, _, _ = make_retriever({"c1": 0.30, "c2": 0.20, "c3": 0.10})
    result = retriever.search("anything", "dense", k=2)
    assert [h.chunk_id for h in result.hits] == ["c1", "c2"]
    assert result.weak_evidence  # 0.30 < default MIN_DENSE_SCORE


def test_hybrid_fuses_dense_and_bm25() -> None:
    # Dense prefers c1, BM25 only matches c2: both must appear with RRF scores.
    retriever, _, _ = make_retriever({"c1": 0.9, "c2": 0.5, "c3": 0.4})
    result = retriever.search("hypertension", "hybrid", k=3)
    scores = {h.chunk_id: h.score for h in result.hits}
    assert scores["c2"] == pytest.approx(1 / 62 + 1 / 61)  # dense rank 2 + bm25 rank 1
    assert scores["c1"] == pytest.approx(1 / 61)
    assert not result.weak_evidence


def test_hybrid_rerank_reorders_and_gates() -> None:
    logits = {"c1": -1.0, "c2": -2.0, "c3": 3.0}
    retriever, _, _ = make_retriever({"c1": 0.9, "c2": 0.5, "c3": 0.4}, logits)
    result = retriever.search("Masern Impfung", "hybrid_rerank", k=2)
    assert result.hits[0].chunk_id == "c3"
    assert result.hits[0].score == pytest.approx(sigmoid(3.0))
    assert not result.weak_evidence

    low = {"c1": -6.0, "c2": -6.0, "c3": -6.0}
    retriever, _, _ = make_retriever({"c1": 0.9, "c2": 0.5, "c3": 0.4}, low)
    assert retriever.search("Masern", "hybrid_rerank", k=2).weak_evidence


def test_unknown_mode_raises() -> None:
    retriever, _, _ = make_retriever({"c1": 0.9, "c2": 0.5, "c3": 0.4})
    with pytest.raises(ValueError):
        retriever.search("x", "sparse", k=2)  # type: ignore[arg-type]
