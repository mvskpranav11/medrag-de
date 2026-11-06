.PHONY: setup up down logs api ui ingest eval test lint format

OLLAMA_MODEL ?= qwen2.5:3b
JUDGE_MODEL ?= qwen2.5:7b

setup:  ## Install all dependencies and pull the local LLM
	uv sync --all-extras
	ollama pull $(OLLAMA_MODEL)
	ollama pull $(JUDGE_MODEL)  # only needed for make eval

up:  ## Start qdrant, api and ui in Docker (Ollama runs on the host)
	docker compose up -d --build
	@echo "UI: http://localhost:8501  API docs: http://localhost:8000/docs"

down:
	docker compose down

logs:
	docker compose logs -f api ui

api:  ## Run the API locally without Docker (needs qdrant running)
	uv run uvicorn medrag.api:app --port 8000

ui:  ## Run the UI locally without Docker
	uv run streamlit run ui/app.py --browser.gatherUsageStats false

ingest:  ## Download, verify, chunk, embed and index the documents
	uv run python -m medrag.ingest

eval:  ## Run the golden-set evaluation and update results
	uv run python -m medrag.evaluate

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
