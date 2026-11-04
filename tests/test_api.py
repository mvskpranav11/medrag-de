from medrag.config import Settings


def test_defaults_are_local_and_bounded() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.max_question_chars > 0
    assert settings.max_context_tokens > 0
    assert settings.request_timeout_s > 0


def test_limits_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MAX_QUESTION_CHARS", "42")
    assert Settings(_env_file=None).max_question_chars == 42
