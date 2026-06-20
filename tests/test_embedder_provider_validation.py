"""v0.13.0: embedding provider Literal narrowing + OpenAI gating tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_knowledgebase.config import Settings


def test_provider_literal_only_accepts_three_values(tmp_path: Path) -> None:
    """Settings.embedding_provider only accepts ollama, sentence-transformers, openai."""
    saves = tmp_path / "saves"
    saves.mkdir()

    for valid in ("ollama", "sentence-transformers", "openai"):
        cfg = Settings(saves_dir=saves, embedding_provider=valid)
        assert cfg.embedding_provider == valid

    for invalid in ("remote", "fastembed", "huggingface"):
        with pytest.raises(ValidationError):
            Settings(saves_dir=saves, embedding_provider=invalid)


def test_default_provider_is_ollama(tmp_path: Path) -> None:
    saves = tmp_path / "saves"
    saves.mkdir()
    cfg = Settings(saves_dir=saves)
    assert cfg.embedding_provider == "ollama"
    assert cfg.embedding_model == "qwen3-embedding:8b"


def test_openai_provider_requires_openai_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Selecting openai when OPENAI_API_KEY is unset raises at embedder build time."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    saves = tmp_path / "saves"
    saves.mkdir()
    cfg = Settings(saves_dir=saves, embedding_provider="openai")

    from agent_knowledgebase.services.embeddings import create_embedder

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_embedder(cfg)


def test_openai_provider_works_when_key_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    saves = tmp_path / "saves"
    saves.mkdir()
    cfg = Settings(
        saves_dir=saves,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    from agent_knowledgebase.services.embeddings import (
        OpenAIEmbedder,
        create_embedder,
    )

    embedder = create_embedder(cfg)
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model_name == "text-embedding-3-small"
