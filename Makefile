.PHONY: setup up down ingest eval test lint format

OLLAMA_MODEL ?= qwen2.5:3b

setup:  ## Install all dependencies and pull the local LLM
	uv sync --all-extras
	ollama pull $(OLLAMA_MODEL)

up:  ## Start the stack (api, qdrant, ui)
	docker compose up -d --build

down:
	docker compose down

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
