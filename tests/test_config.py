"""Tests for agent_knowledgebase.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.config import Settings


class TestDefaults:
    """Default values load correctly without any env vars."""

    def test_default_db_path(self) -> None:
        cfg = Settings()
        assert cfg.db_path == Path("~/.agent-kb/knowledgebase.db")

    def test_default_vectorstore(self) -> None:
        cfg = Settings()
        assert cfg.vectorstore == "chromadb"

    def test_default_chroma_path(self) -> None:
        cfg = Settings()
        assert cfg.chroma_path == Path("~/.agent-kb/chroma")

    def test_default_embedding_provider(self) -> None:
        cfg = Settings()
        assert cfg.embedding_provider == "sentence-transformers"

    def test_default_embedding_model(self) -> None:
        cfg = Settings()
        assert cfg.embedding_model == "all-MiniLM-L6-v2"

    def test_default_chunk_size(self) -> None:
        cfg = Settings()
        assert cfg.chunk_size == 512

    def test_default_chunk_overlap(self) -> None:
        cfg = Settings()
        assert cfg.chunk_overlap == 64

    def test_optional_fields_are_none(self) -> None:
        cfg = Settings()
        assert cfg.pinecone_api_key is None
        assert cfg.pinecone_index is None
        assert cfg.pinecone_environment is None
        assert cfg.openai_api_key is None
        assert cfg.export_path is None


class TestEnvOverrides:
    """Environment variable overrides are picked up."""

    def test_override_db_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_KB_DB_PATH", "/tmp/custom.db")
        cfg = Settings()
        assert cfg.db_path == Path("/tmp/custom.db")

    def test_override_vectorstore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_KB_VECTORSTORE", "pinecone")
        cfg = Settings()
        assert cfg.vectorstore == "pinecone"

    def test_override_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_KB_CHUNK_SIZE", "1024")
        cfg = Settings()
        assert cfg.chunk_size == 1024

    def test_override_chunk_overlap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_KB_CHUNK_OVERLAP", "128")
        cfg = Settings()
        assert cfg.chunk_overlap == 128

    def test_override_embedding_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_KB_EMBEDDING_PROVIDER", "openai")
        cfg = Settings()
        assert cfg.embedding_provider == "openai"

    def test_override_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_KB_EMBEDDING_MODEL", "text-embedding-3-small")
        cfg = Settings()
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_override_export_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_KB_EXPORT_PATH", "/tmp/export")
        cfg = Settings()
        assert cfg.export_path == Path("/tmp/export")


class TestPathExpansion:
    """Tilde paths are expanded via resolve_paths."""

    def test_tilde_expanded_in_db_path(self) -> None:
        cfg = Settings().resolve_paths()
        assert "~" not in str(cfg.db_path)
        assert cfg.db_path.is_absolute()

    def test_tilde_expanded_in_chroma_path(self) -> None:
        cfg = Settings().resolve_paths()
        assert "~" not in str(cfg.chroma_path)
        assert cfg.chroma_path.is_absolute()

    def test_none_export_path_stays_none(self) -> None:
        cfg = Settings().resolve_paths()
        assert cfg.export_path is None

    def test_set_export_path_expanded(self) -> None:
        cfg = Settings(export_path=Path("~/my-export")).resolve_paths()
        assert "~" not in str(cfg.export_path)
        assert cfg.export_path.is_absolute()


class TestFixtures:
    """Verify the shared fixtures from conftest work."""

    def test_tmp_db_path_is_in_tmp(self, tmp_db_path: Path) -> None:
        assert tmp_db_path.name == "test_knowledgebase.db"

    def test_test_config_uses_tmp_paths(self, test_config: Settings, tmp_path: Path) -> None:
        assert test_config.db_path.parent == tmp_path
        assert test_config.chroma_path.parent == tmp_path
        assert test_config.export_path is not None
        assert test_config.export_path.parent == tmp_path
