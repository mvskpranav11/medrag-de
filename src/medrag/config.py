"""Application settings, read from environment variables or a local `.env` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Data
    manifest_path: Path = Path("data/manifest.yaml")
    raw_dir: Path = Path("data/raw")

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "medrag"

    # Models (sized for an 8 GB Apple Silicon machine)
    embedding_model: str = "BAAI/bge-m3"
    # Small multilingual cross-encoder (118M params). bge-reranker-v2-m3 does not fit next
    # to the LLM in 8 GB; set RERANKER_MODEL=BAAI/bge-reranker-v2-m3 on bigger machines.
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Query-time embedder + re-ranker run on CPU by default so the GPU stays free for the
    # LLM (on 8 GB Apple Silicon all three do not fit in GPU memory together).
    device: Literal["auto", "cpu", "mps", "cuda"] = "cpu"
    # Ingestion runs without the LLM, so it can use the GPU.
    ingest_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"

    # "Not found" thresholds (calibrated scores only; see retrieval.search)
    min_dense_score: float = 0.60
    min_rerank_score: float = 0.50

    # Chunking (characters)
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # LLM: local Ollama by default; a cloud provider is opt-in.
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Evaluation judge (RAGAS). Defaults to the same local model.
    judge_provider: Literal["ollama", "openai"] = "ollama"
    judge_model: str = "qwen2.5:3b"

    # Limits, enforced in code
    max_question_chars: int = 500
    max_context_tokens: int = 3000
    max_top_k: int = 20
    request_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
