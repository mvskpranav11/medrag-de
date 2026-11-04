"""FastAPI service: GET /health and POST /query.

Run locally with `uvicorn medrag.api:app`. Models load once at startup, and queries
run one at a time, which keeps memory bounded on small machines.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from medrag.config import Settings, get_settings
from medrag.generate import LLM, Answer, answer_question, make_llm
from medrag.retrieval import Mode, Retriever

DISCLAIMER = "Demo only. Not medical advice."


@dataclass
class Pipeline:
    settings: Settings
    retriever: Retriever
    llm: LLM
    lock: threading.Lock = field(default_factory=threading.Lock)

    def answer(self, question: str, mode: Mode, top_k: int) -> Answer:
        with self.lock:  # one query at a time: the models share 8 GB of memory
            return answer_question(question, mode, top_k, self.retriever, self.llm, self.settings)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Ab welchem Blutdruck soll behandelt werden?"])
    mode: Mode = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1)

    @field_validator("question")
    @classmethod
    def check_question(cls, value: str) -> str:
        limit = get_settings().max_question_chars
        if not value.strip():
            raise ValueError("question must not be blank")
        if len(value) > limit:
            raise ValueError(f"question must be at most {limit} characters")
        return value.strip()

    @field_validator("top_k")
    @classmethod
    def check_top_k(cls, value: int) -> int:
        limit = get_settings().max_top_k
        if value > limit:
            raise ValueError(f"top_k must be at most {limit}")
        return value


class CitationOut(BaseModel):
    doc_id: str
    title: str
    page: int
    snippet: str
    score: float


class Latency(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    answer: str
    found: bool
    language: Literal["de", "en"]
    mode: Mode
    citations: list[CitationOut]
    latency: Latency
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant: bool
    llm: bool
    chunks: int
    llm_provider: str


def build_pipeline(settings: Settings) -> Pipeline:
    retriever = Retriever(settings)
    retriever.warm_up()
    return Pipeline(settings, retriever, make_llm(settings))


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or await asyncio.to_thread(build_pipeline, get_settings())
        yield

    app = FastAPI(title="MedRAG-DE", description=DISCLAIMER, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        pipe: Pipeline = request.app.state.pipeline
        try:
            chunks, qdrant_ok = pipe.retriever.count(), True
        except Exception:  # any connection problem means "not healthy", not a crash
            chunks, qdrant_ok = 0, False
        llm_ok = pipe.llm.ping()
        status = "ok" if qdrant_ok and llm_ok and chunks > 0 else "degraded"
        return HealthResponse(
            status=status,
            qdrant=qdrant_ok,
            llm=llm_ok,
            chunks=chunks,
            llm_provider=pipe.settings.llm_provider,
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        pipe: Pipeline = request.app.state.pipeline
        timeout = pipe.settings.request_timeout_s
        work = asyncio.to_thread(pipe.answer, body.question, body.mode, body.top_k)
        try:
            result = await asyncio.wait_for(work, timeout=timeout)
        except TimeoutError as exc:
            raise HTTPException(504, f"Request took longer than {timeout:.0f} s.") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"LLM backend error: {exc.__class__.__name__}") from exc
        return QueryResponse(
            answer=result.answer,
            found=result.found,
            language=result.language,
            mode=body.mode,
            citations=[CitationOut(**vars(c)) for c in result.citations],
            latency=Latency(
                retrieval_ms=round(result.retrieval_ms, 1),
                generation_ms=round(result.generation_ms, 1),
                total_ms=round(result.retrieval_ms + result.generation_ms, 1),
            ),
        )

    return app


app = create_app()
