import time

import pytest
from fastapi.testclient import TestClient

from medrag.api import Pipeline, create_app
from medrag.config import Settings
from medrag.retrieval import Hit, Retrieval

HITS = [Hit("c1", "who-pa", "WHO PA", "en", 3, "Adults: 150-300 minutes per week.", 0.9)]


class FakeRetriever:
    def __init__(self, chunks: int = 1153) -> None:
        self.chunks = chunks

    def search(self, query: str, mode: str, k: int) -> Retrieval:
        return Retrieval(HITS, weak_evidence=False)

    def count(self) -> int:
        return self.chunks


class FakeLLM:
    def __init__(self, reply: str = "150-300 minutes [1].", delay: float = 0.0) -> None:
        self.reply, self.delay = reply, delay

    def complete(self, system: str, user: str) -> str:
        time.sleep(self.delay)
        return self.reply

    def ping(self) -> bool:
        return True


def client_for(llm: FakeLLM | None = None, timeout: float = 5.0, chunks: int = 1153):
    settings = Settings(_env_file=None, request_timeout_s=timeout)
    pipeline = Pipeline(settings, FakeRetriever(chunks), llm or FakeLLM())  # type: ignore[arg-type]
    return TestClient(create_app(pipeline))


def test_defaults_are_local_and_bounded() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.max_question_chars > 0
    assert settings.max_context_tokens > 0
    assert settings.request_timeout_s > 0


def test_limits_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_QUESTION_CHARS", "42")
    assert Settings(_env_file=None).max_question_chars == 42


def test_health_reports_components() -> None:
    with client_for() as client:
        body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "qdrant": True,
        "llm": True,
        "chunks": 1153,
        "llm_provider": "ollama",
    }


def test_health_degraded_when_index_empty() -> None:
    with client_for(chunks=0) as client:
        assert client.get("/health").json()["status"] == "degraded"


def test_query_returns_answer_citations_and_latency() -> None:
    with client_for() as client:
        response = client.post("/query", json={"question": "How much activity?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["mode"] == "hybrid_rerank"
    assert body["answer"] == "150-300 minutes [WHO PA, p. 3]."
    assert body["citations"][0] == {
        "doc_id": "who-pa",
        "title": "WHO PA",
        "page": 3,
        "snippet": "Adults: 150-300 minutes per week.",
        "score": 0.9,
    }
    assert set(body["latency"]) == {"retrieval_ms", "generation_ms", "total_ms"}
    assert "not medical advice" in body["disclaimer"].lower()


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 501},
        {"question": "ok?", "top_k": 0},
        {"question": "ok?", "top_k": 21},
        {"question": "ok?", "mode": "sparse"},
    ],
)
def test_query_rejects_invalid_input(payload: dict) -> None:
    with client_for() as client:
        assert client.post("/query", json=payload).status_code == 422


def test_query_times_out() -> None:
    with client_for(FakeLLM(delay=0.5), timeout=0.1) as client:
        assert client.post("/query", json={"question": "How much activity?"}).status_code == 504
