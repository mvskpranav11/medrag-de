"""Answer generation: prompt, citations, language handling and the "not found" rule."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from medrag.config import Settings
from medrag.retrieval import MODES, Hit, Mode, Retriever

Language = Literal["de", "en"]

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE: dict[Language, str] = {
    "de": "Dazu habe ich in den Dokumenten nichts gefunden.",
    "en": "Not found in the documents.",
}
LANGUAGE_NAME: dict[Language, str] = {"de": "German", "en": "English"}
SNIPPET_CHARS = 300
OLLAMA_NUM_CTX = 8192  # room for MAX_CONTEXT_TOKENS plus prompt and answer

_DE_WORDS = frozenset(
    "der die das und ist sind nicht ein eine einer wie welche welcher wann wer wo warum "
    "sollte sollten soll werden wird für mit bei von zu zum zur im den dem des kann können "
    "auf oder auch nach über ab wie viel viele empfohlen empfiehlt".split()
)
_EN_WORDS = frozenset(
    "the and is are not a an how what which when who where why should be will for with "
    "at of to in does do can on or also after about from many much recommended".split()
)
# One or more adjacent markers such as [1], [2, 3], [1][2], [1] [2] or [1], [2]
_MARKER = r"\[\d+(?:\s*,\s*\d+)*\]"
_CITATION_GROUP = re.compile(rf"{_MARKER}(?:[\s,;]*{_MARKER})*")

SYSTEM_PROMPT = """You answer questions about medical guideline documents.
Rules:
1. Use only the numbered sources. Do not use outside knowledge.
2. After every sentence with a fact, cite its source number in square brackets, e.g. [1] or [2][3].
3. Write the answer in {language}, even if the sources are in another language.
4. Be brief: at most 4 sentences.
5. If the sources do not contain the answer, reply with exactly: NOT_FOUND"""


@dataclass(frozen=True)
class Citation:
    doc_id: str
    title: str
    page: int
    snippet: str
    score: float


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    found: bool
    language: Language
    retrieval_ms: float
    generation_ms: float


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...

    def ping(self) -> bool: ...


class OllamaLLM:
    def __init__(self, url: str, model: str, timeout: float) -> None:
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def ping(self) -> bool:
        """True if Ollama is reachable and the model has been pulled."""
        try:
            response = httpx.get(f"{self.url}/api/tags", timeout=3)
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        names = {m["name"] for m in response.json().get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def complete(self, system: str, user: str) -> str:
        response = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class OpenAILLM:
    """Any OpenAI-compatible chat API. Only used when LLM_PROVIDER=openai."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self.model, self.timeout = model, timeout

    def ping(self) -> bool:
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def complete(self, system: str, user: str) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def make_llm(settings: Settings, provider: str | None = None, model: str | None = None) -> LLM:
    provider = provider or settings.llm_provider
    if provider == "ollama":
        return OllamaLLM(
            settings.ollama_url, model or settings.ollama_model, settings.request_timeout_s
        )
    if provider == "openai":
        return OpenAILLM(
            settings.openai_base_url,
            settings.openai_api_key,
            model or settings.openai_model,
            settings.request_timeout_s,
        )
    raise ValueError(f"Unknown LLM provider {provider!r}")


def detect_language(text: str) -> Language:
    """Tiny stopword heuristic; umlauts and ß count double for German. Ties go to English."""
    lowered = text.lower()
    tokens = re.findall(r"\w+", lowered)
    german = sum(t in _DE_WORDS for t in tokens) + 2 * len(re.findall(r"[äöüß]", lowered))
    english = sum(t in _EN_WORDS for t in tokens)
    return "de" if german > english else "en"


def validate_question(question: str, settings: Settings) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")
    if len(question) > settings.max_question_chars:
        raise ValueError(f"Question is longer than {settings.max_question_chars} characters.")
    return question


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 characters per token), good enough for a budget limit."""
    return math.ceil(len(text) / 4)


def select_context(hits: list[Hit], max_tokens: int) -> list[Hit]:
    """Keep hits in rank order until the token budget would be exceeded."""
    selected: list[Hit] = []
    used = 0
    for hit in hits:
        cost = estimate_tokens(hit.text)
        if used + cost > max_tokens:
            break
        selected.append(hit)
        used += cost
    return selected


def build_user_prompt(question: str, sources: list[Hit]) -> str:
    blocks = [f"[{i}] {hit.title}, p. {hit.page}\n{hit.text}" for i, hit in enumerate(sources, 1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def render_citations(text: str, sources: list[Hit]) -> tuple[str, list[int]]:
    """Replace [n] markers with [title, p. X]; drop invalid numbers.

    Sources from the same document page are merged into one citation. Returns the
    numbers of the cited sources (one per page, first occurrence wins).
    """
    canonical: dict[tuple[str, int], int] = {}
    for number, hit in enumerate(sources, 1):
        canonical.setdefault((hit.doc_id, hit.page), number)
    used: list[int] = []

    def replace_group(match: re.Match[str]) -> str:
        numbers = [int(n) for n in re.findall(r"\d+", match.group(0))]
        cited = [
            canonical[(sources[n - 1].doc_id, sources[n - 1].page)]
            for n in numbers
            if 1 <= n <= len(sources)
        ]
        cited = list(dict.fromkeys(cited))
        used.extend(n for n in cited if n not in used)
        return "".join(f"[{sources[n - 1].title}, p. {sources[n - 1].page}]" for n in cited)

    rendered = _CITATION_GROUP.sub(replace_group, text)
    return re.sub(r"[ \t]{2,}", " ", rendered).strip(), used


def to_citation(hit: Hit) -> Citation:
    snippet = hit.text if len(hit.text) <= SNIPPET_CHARS else hit.text[:SNIPPET_CHARS] + "…"
    return Citation(hit.doc_id, hit.title, hit.page, snippet, round(hit.score, 4))


def not_found(language: Language, retrieval_ms: float, generation_ms: float) -> Answer:
    return Answer(NOT_FOUND_MESSAGE[language], [], False, language, retrieval_ms, generation_ms)


def answer_question(
    question: str,
    mode: Mode,
    top_k: int,
    retriever: Retriever,
    llm: LLM,
    settings: Settings,
) -> Answer:
    question = validate_question(question, settings)
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}")
    if not 1 <= top_k <= settings.max_top_k:
        raise ValueError(f"top_k must be between 1 and {settings.max_top_k}.")
    language = detect_language(question)

    start = time.perf_counter()
    retrieval = retriever.search(question, mode, top_k)
    retrieval_ms = (time.perf_counter() - start) * 1000
    sources = select_context(retrieval.hits, settings.max_context_tokens)
    if retrieval.weak_evidence or not sources:
        return not_found(language, retrieval_ms, 0.0)

    start = time.perf_counter()
    system = SYSTEM_PROMPT.format(language=LANGUAGE_NAME[language])
    raw = llm.complete(system, build_user_prompt(question, sources))
    generation_ms = (time.perf_counter() - start) * 1000

    text, used = render_citations(raw, sources)
    if NOT_FOUND_TOKEN in raw or not used:
        return not_found(language, retrieval_ms, generation_ms)
    citations = [to_citation(sources[n - 1]) for n in used]
    return Answer(text, citations, True, language, retrieval_ms, generation_ms)
