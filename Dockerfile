# One image for both the API and the UI (they differ only in the start command).
FROM python:3.11.14-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home app && mkdir -p /home/app/.cache/huggingface \
    && chown -R app /home/app/.cache
WORKDIR /app

# Dependencies first, so code changes don't reinstall torch.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ml --no-install-project

COPY src ./src
COPY ui ./ui
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ml

USER app
EXPOSE 8000 8501
CMD ["uvicorn", "medrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
