"""Tests for agent_knowledgebase.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.config import Settings, sanitize_kb_dir_name


class TestSavesDir:
    """saves_dir is required and validated."""

    def test_saves_dir_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings() without AGENT_KB_SAVES_DIR raises."""
        monkeypatch.delenv("AGENT_KB_SAVES_DIR", raising=False)
        with pytest.raises(Exception):
            Settings()

    def test_saves_dir_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        saves = tmp_path / "env_saves"
        saves.mkdir()
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        cfg = Settings()
        assert cfg.saves_dir == saves

    def test_resolve_paths_fails_if_dir_missing(self, tmp_path: Path) -> None:
        """Missing saves_dir must raise loudly — no silent auto-create."""
        cfg = Settings(saves_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="AGENT_KB_SAVES_DIR"):
            cfg.resolve_paths()

    def test_resolve_paths_rejects_file_as_saves_dir(self, tmp_path: Path) -> None:
        """If the path exists as a file, raise instead of clobbering it."""
        not_a_dir = tmp_path / "iam_a_file"
        not_a_dir.write_text("hi")
        with pytest.raises(NotADirectoryError, match="AGENT_KB_SAVES_DIR"):
            Settings(saves_dir=not_a_dir).resolve_paths()

    def test_resolve_paths_succeeds_for_existing_dir(self, tmp_path: Path) -> None:
        saves = tmp_path / "ok"
        saves.mkdir()
        resolved = Settings(saves_dir=saves).resolve_paths()
        assert resolved.saves_dir.is_absolute()
        assert resolved.saves_dir.is_dir()


class TestDefaults:
    """Default values for fields other than saves_dir."""

    def test_default_vectorstore(self, test_config: Settings) -> None:
        assert test_config.vectorstore == "chromadb"

    def test_default_embedding_provider(self, test_config: Settings) -> None:
        assert test_config.embedding_provider == "sentence-transformers"

    def test_default_embedding_model(self, test_config: Settings) -> None:
        assert test_config.embedding_model == "all-MiniLM-L6-v2"

    def test_default_chunk_size(self, test_config: Settings) -> None:
        assert test_config.chunk_size == 512

    def test_default_chunk_overlap(self, test_config: Settings) -> None:
        assert test_config.chunk_overlap == 64

    def test_optional_fields_are_none(self, test_config: Settings) -> None:
        assert test_config.pinecone_api_key is None
        assert test_config.pinecone_index is None
        assert test_config.pinecone_environment is None
        assert test_config.openai_api_key is None

    def test_default_chunk_token_encoding(self, test_config: Settings) -> None:
        assert test_config.chunk_token_encoding == "cl100k_base"

    def test_default_query_default_top_k(self, test_config: Settings) -> None:
        assert test_config.query_default_top_k == 10

    def test_default_query_hybrid_vector_weight(self, test_config: Settings) -> None:
        assert test_config.query_hybrid_vector_weight == 0.7

    def test_default_query_hybrid_fts_weight(self, test_config: Settings) -> None:
        assert test_config.query_hybrid_fts_weight == 0.3

    def test_default_query_hybrid_fetch_multiplier(self, test_config: Settings) -> None:
        assert test_config.query_hybrid_fetch_multiplier == 2

    def test_default_ingest_excluded_dirs(self, test_config: Settings) -> None:
        assert test_config.ingest_excluded_dirs == [
            "__pycache__", "node_modules", ".git", ".venv",
            ".mypy_cache", ".pytest_cache", "dist", "build",
            "venv", ".tox", ".ruff_cache", ".eggs",
            ".idea", ".vscode", ".hg", ".svn",
        ]


class TestEnvOverrides:
    """Environment variable overrides are picked up."""

    def test_override_vectorstore(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("AGENT_KB_VECTORSTORE", "pinecone")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.vectorstore == "pinecone"

    def test_override_chunk_size(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("AGENT_KB_CHUNK_SIZE", "1024")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.chunk_size == 1024

    def test_override_chunk_overlap(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("AGENT_KB_CHUNK_OVERLAP", "128")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.chunk_overlap == 128

    def test_override_embedding_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENT_KB_EMBEDDING_PROVIDER", "openai")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.embedding_provider == "openai"

    def test_override_embedding_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENT_KB_EMBEDDING_MODEL", "text-embedding-3-small")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_override_export_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("AGENT_KB_EXPORT_PATH", "/tmp/export")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.export_path == Path("/tmp/export")

    def test_override_chunk_token_encoding(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENT_KB_CHUNK_TOKEN_ENCODING", "o200k_base")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.chunk_token_encoding == "o200k_base"

    def test_override_query_default_top_k(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENT_KB_QUERY_DEFAULT_TOP_K", "25")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.query_default_top_k == 25

    def test_override_query_hybrid_weights(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENT_KB_QUERY_HYBRID_VECTOR_WEIGHT", "0.6")
        monkeypatch.setenv("AGENT_KB_QUERY_HYBRID_FTS_WEIGHT", "0.4")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.query_hybrid_vector_weight == 0.6
        assert cfg.query_hybrid_fts_weight == 0.4

    def test_override_query_hybrid_fetch_multiplier(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENT_KB_QUERY_HYBRID_FETCH_MULTIPLIER", "3")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.query_hybrid_fetch_multiplier == 3

    def test_override_ingest_excluded_dirs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AGENT_KB_INGEST_EXCLUDED_DIRS", "foo, bar,baz")
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.ingest_excluded_dirs == ["foo", "bar", "baz"]


class TestPathExpansion:
    """Tilde paths are expanded via resolve_paths."""

    def test_tilde_expanded_in_saves_dir(self, tmp_path: Path) -> None:
        saves = tmp_path / "tilde_test"
        saves.mkdir()
        cfg = Settings(saves_dir=saves).resolve_paths()
        assert "~" not in str(cfg.saves_dir)
        assert cfg.saves_dir.is_absolute()

    def test_none_export_path_stays_none(self, tmp_path: Path) -> None:
        saves = tmp_path / "ep_test"
        saves.mkdir()
        cfg = Settings(saves_dir=saves).resolve_paths()
        assert cfg.export_path is None

    def test_set_export_path_expanded(self, tmp_path: Path) -> None:
        saves = tmp_path / "ep2_test"
        saves.mkdir()
        cfg = Settings(saves_dir=saves, export_path=Path("~/my-export")).resolve_paths()
        assert "~" not in str(cfg.export_path)
        assert cfg.export_path.is_absolute()


class TestPerKBPaths:
    """Per-KB path helpers derive from saves_dir."""

    def test_knowledgebases_dir_is_saves_dir(self, tmp_path: Path) -> None:
        """Each KB is placed directly under saves_dir, not under a nested folder."""
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.knowledgebases_dir == tmp_path

    def test_kb_data_dir(self, tmp_path: Path) -> None:
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.kb_data_dir("my-kb") == tmp_path / "my-kb"

    def test_kb_db_path(self, tmp_path: Path) -> None:
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.kb_db_path("my-kb") == tmp_path / "my-kb" / "knowledgebase.db"

    def test_kb_chroma_path(self, tmp_path: Path) -> None:
        cfg = Settings(saves_dir=tmp_path)
        assert cfg.kb_chroma_path("my-kb") == tmp_path / "my-kb" / "chroma"


class TestSanitizeKBDirName:
    """sanitize_kb_dir_name produces safe directory names."""

    def test_basic(self) -> None:
        assert sanitize_kb_dir_name("My Research KB") == "my-research-kb"

    def test_special_chars_stripped(self) -> None:
        assert sanitize_kb_dir_name("Hello! @World#") == "hello-world"

    def test_underscores_become_hyphens(self) -> None:
        assert sanitize_kb_dir_name("foo_bar_baz") == "foo-bar-baz"

    def test_multiple_spaces_collapsed(self) -> None:
        assert sanitize_kb_dir_name("  hello   world  ") == "hello-world"

    def test_empty_returns_unnamed(self) -> None:
        assert sanitize_kb_dir_name("") == "unnamed"
        assert sanitize_kb_dir_name("!!!") == "unnamed"

    def test_already_clean(self) -> None:
        assert sanitize_kb_dir_name("clean-name") == "clean-name"


class TestChunkOverlapValidation:
    """chunk_overlap must be < chunk_size."""

    def test_overlap_too_large_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            Settings(saves_dir=tmp_path, chunk_overlap=512, chunk_size=512)


class TestHybridWeightValidation:
    """query_hybrid_vector_weight + query_hybrid_fts_weight must sum to 1.0 ±1e-6."""

    def test_weights_summing_to_one_pass(self, tmp_path: Path) -> None:
        cfg = Settings(
            saves_dir=tmp_path,
            query_hybrid_vector_weight=0.6,
            query_hybrid_fts_weight=0.4,
        )
        assert cfg.query_hybrid_vector_weight == 0.6
        assert cfg.query_hybrid_fts_weight == 0.4

    def test_weights_not_summing_to_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="query_hybrid"):
            Settings(
                saves_dir=tmp_path,
                query_hybrid_vector_weight=0.8,
                query_hybrid_fts_weight=0.3,
            )

    def test_weights_within_epsilon_pass(self, tmp_path: Path) -> None:
        # Floating-point safe: 0.7 + 0.3 is sometimes 0.9999999...
        Settings(
            saves_dir=tmp_path,
            query_hybrid_vector_weight=0.7,
            query_hybrid_fts_weight=0.3,
        )


class TestFixtures:
    """Verify the shared fixtures from conftest work."""

    def test_test_config_has_saves_dir(self, test_config: Settings) -> None:
        assert test_config.saves_dir is not None
        assert test_config.saves_dir.is_dir()

    def test_test_config_export_path(self, test_config: Settings) -> None:
        assert test_config.export_path is not None
