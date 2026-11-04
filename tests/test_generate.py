import pytest

from medrag.config import Settings
from medrag.generate import (
    NOT_FOUND_MESSAGE,
    answer_question,
    detect_language,
    select_context,
)
from medrag.retrieval import Hit, Retrieval

SETTINGS = Settings(_env_file=None)
HITS = [
    Hit("c1", "who-pa", "WHO PA", "en", 3, "Adults should do 150-300 minutes per week.", 0.8),
    Hit("c2", "who-htn", "WHO HTN", "en", 7, "Start treatment at 140/90 mmHg.", 0.7),
]


class FakeRetriever:
    def __init__(self, hits: list[Hit], weak: bool = False) -> None:
        self.result = Retrieval(hits, weak_evidence=weak)

    def search(self, query: str, mode: str, k: int) -> Retrieval:
        return self.result


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def ask(question: str, llm: FakeLLM, hits=HITS, weak: bool = False):
    return answer_question(question, "hybrid", 5, FakeRetriever(hits, weak), llm, SETTINGS)


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("Wie viel Bewegung wird für Erwachsene empfohlen?", "de"),
        ("Ab welchem Blutdruck soll behandelt werden?", "de"),
        ("How much physical activity is recommended for adults?", "en"),
        ("When should treatment start?", "en"),
    ],
)
def test_detect_language(question: str, language: str) -> None:
    assert detect_language(question) == language


def test_citations_rendered_and_structured() -> None:
    answer = ask("How much activity?", FakeLLM("150-300 minutes [1]. Treat at 140/90 [2]."))
    assert answer.found
    assert answer.answer == "150-300 minutes [WHO PA, p. 3]. Treat at 140/90 [WHO HTN, p. 7]."
    assert [(c.title, c.page) for c in answer.citations] == [("WHO PA", 3), ("WHO HTN", 7)]
    assert answer.citations[0].snippet.startswith("Adults should")


def test_invalid_citation_numbers_are_dropped() -> None:
    answer = ask("How much activity?", FakeLLM("150-300 minutes [1][9]."))
    assert answer.answer == "150-300 minutes [WHO PA, p. 3]."
    assert len(answer.citations) == 1


def test_same_page_citations_are_merged() -> None:
    same_page = HITS + [Hit("c3", "who-pa", "WHO PA", "en", 3, "More text from page 3.", 0.6)]
    answer = ask("How much activity?", FakeLLM("150-300 minutes [1] [3]. Also [2, 3]."), same_page)
    assert answer.answer == "150-300 minutes [WHO PA, p. 3]. Also [WHO HTN, p. 7][WHO PA, p. 3]."
    assert [(c.title, c.page) for c in answer.citations] == [("WHO PA", 3), ("WHO HTN", 7)]


def test_weak_evidence_refuses_without_calling_llm() -> None:
    llm = FakeLLM("should not be used [1]")
    answer = ask("Wie behandelt man Migräne bei Kindern?", llm, weak=True)
    assert not answer.found
    assert answer.answer == NOT_FOUND_MESSAGE["de"]
    assert llm.calls == []


def test_llm_not_found_token_refuses() -> None:
    answer = ask("What is the dose of drug X?", FakeLLM("NOT_FOUND"))
    assert not answer.found
    assert answer.answer == NOT_FOUND_MESSAGE["en"]
    assert answer.citations == []


def test_answer_without_citation_is_refused() -> None:
    answer = ask("How much activity?", FakeLLM("Adults should exercise a lot."))
    assert not answer.found


def test_prompt_asks_for_question_language() -> None:
    llm = FakeLLM("150–300 Minuten pro Woche [1].")
    answer = ask("Wie viel Bewegung wird für Erwachsene empfohlen?", llm)
    system, user = llm.calls[0]
    assert "Write the answer in German" in system
    assert "[1] WHO PA, p. 3" in user
    assert answer.language == "de"


def test_limits_are_enforced() -> None:
    llm = FakeLLM("x [1]")
    with pytest.raises(ValueError):
        ask("x" * (SETTINGS.max_question_chars + 1), llm)
    with pytest.raises(ValueError):
        ask("   ", llm)
    with pytest.raises(ValueError):
        answer_question("ok?", "hybrid", 0, FakeRetriever(HITS), llm, SETTINGS)


def test_select_context_respects_token_budget() -> None:
    long_hits = [Hit(f"c{i}", "d", "D", "en", i, "x" * 400, 0.5) for i in range(10)]  # 100 tok
    assert len(select_context(long_hits, max_tokens=350)) == 3
