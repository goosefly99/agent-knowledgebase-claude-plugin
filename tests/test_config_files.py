"""Tests for config file path resolution and JSON source."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest


class TestUserConfigPathResolution:
    """~/.agent-kb/config.json, overridable via AGENT_KB_USER_CONFIG."""

    def test_default_user_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_knowledgebase.config_files import resolve_user_config_path
        monkeypatch.delenv("AGENT_KB_USER_CONFIG", raising=False)
        path = resolve_user_config_path()
        assert path.name == "config.json"
        assert path.parent.name == ".agent-kb"
        assert path.parent.parent == Path.home()

    def test_env_override_user_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config_files import resolve_user_config_path
        custom = tmp_path / "custom.json"
        monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(custom))
        assert resolve_user_config_path() == custom


class TestProjectConfigPathResolution:
    """./.agent-kb/config.json with a chain of <project_root> fallbacks."""

    def test_env_override_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config_files import resolve_project_config_path
        custom = tmp_path / "proj.json"
        monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(custom))
        # Other vars should be ignored when explicit override set
        monkeypatch.setenv("AGENT_KB_PROJECT_DIR", str(tmp_path / "nope"))
        assert resolve_project_config_path() == custom

    def test_agent_kb_project_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config_files import resolve_project_config_path
        monkeypatch.delenv("AGENT_KB_PROJECT_CONFIG", raising=False)
        monkeypatch.setenv("AGENT_KB_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/should/not/win")
        monkeypatch.setenv("PWD", "/should/not/win")
        assert resolve_project_config_path() == tmp_path / ".agent-kb" / "config.json"

    def test_claude_project_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config_files import resolve_project_config_path
        monkeypatch.delenv("AGENT_KB_PROJECT_CONFIG", raising=False)
        monkeypatch.delenv("AGENT_KB_PROJECT_DIR", raising=False)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
        monkeypatch.setenv("PWD", "/should/not/win")
        assert resolve_project_config_path() == tmp_path / ".agent-kb" / "config.json"

    def test_pwd_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config_files import resolve_project_config_path
        monkeypatch.delenv("AGENT_KB_PROJECT_CONFIG", raising=False)
        monkeypatch.delenv("AGENT_KB_PROJECT_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setenv("PWD", str(tmp_path))
        assert resolve_project_config_path() == tmp_path / ".agent-kb" / "config.json"

    def test_cwd_last_resort(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config_files import resolve_project_config_path
        monkeypatch.delenv("AGENT_KB_PROJECT_CONFIG", raising=False)
        monkeypatch.delenv("AGENT_KB_PROJECT_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.delenv("PWD", raising=False)
        original_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            result = resolve_project_config_path()
            assert result.parent.parent.resolve() == tmp_path.resolve()
        finally:
            os.chdir(original_cwd)


def _make_source(path: Path):
    """Instantiate the source against a target file path."""
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.config_files import NestedJsonConfigSettingsSource
    return NestedJsonConfigSettingsSource(Settings, path=path)


def _load(path: Path) -> dict[str, Any]:
    return _make_source(path)()


class TestNestedJsonConfigSettingsSourceLoad:
    def test_missing_file_returns_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("WARNING", logger="agent_knowledgebase.config")
        result = _load(tmp_path / "no_such.json")
        assert result == {}
        assert any("no_such.json" in m for m in caplog.messages)

    def test_empty_dict_returns_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("WARNING", logger="agent_knowledgebase.config")
        f = tmp_path / "cfg.json"
        f.write_text("{}")
        result = _load(f)
        assert result == {}
        assert any(str(f) in m for m in caplog.messages)

    def test_flattening(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps({
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "chunk": {"size": 256},
            "query": {"hybrid": {"vector_weight": 0.6}},
        }))
        result = _load(f)
        assert result["embedding_provider"] == "openai"
        assert result["embedding_model"] == "text-embedding-3-small"
        assert result["chunk_size"] == 256
        assert result["query_hybrid_vector_weight"] == 0.6

    def test_flat_keys_also_accepted(self, tmp_path: Path) -> None:
        """Top-level keys like `vectorstore` and `export_path` pass through unchanged."""
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps({"vectorstore": "pinecone", "export_path": "/tmp/out"}))
        result = _load(f)
        assert result["vectorstore"] == "pinecone"
        assert result["export_path"] == "/tmp/out"

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.json"
        f.write_text("{not json")
        with pytest.raises(ValueError, match=re.escape(str(f))):
            _load(f)

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps({"embedding": {"typo_key": "x"}}))
        with pytest.raises(ValueError, match="embedding.typo_key"):
            _load(f)

    @pytest.mark.parametrize(
        "forbidden_key,forbidden_container",
        [
            ("openai_api_key", {"openai_api_key": "sk-..."}),
            ("pinecone_api_key", {"pinecone": {"api_key": "..."}}),
            ("saves_dir", {"saves_dir": "/some/path"}),
        ],
    )
    def test_forbidden_keys_rejected(
        self, tmp_path: Path, forbidden_key: str, forbidden_container: dict
    ) -> None:
        f = tmp_path / "cfg.json"
        f.write_text(json.dumps(forbidden_container))
        with pytest.raises(ValueError, match=forbidden_key):
            _load(f)


class TestLayeringPrecedence:
    """default < user JSON < project JSON < env."""

    def _write(self, path: Path, content: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))

    def test_default_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config import Settings
        saves = tmp_path / "saves"
        saves.mkdir()
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(tmp_path / "no_user.json"))
        monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(tmp_path / "no_proj.json"))
        cfg = Settings()
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.chunk_size == 512

    def test_user_file_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config import Settings
        saves = tmp_path / "saves"
        saves.mkdir()
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        user_file = tmp_path / "user.json"
        self._write(user_file, {"embedding": {"model": "from-user"}, "chunk": {"size": 128}})
        monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(user_file))
        monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(tmp_path / "no_proj.json"))
        cfg = Settings()
        assert cfg.embedding_model == "from-user"
        assert cfg.chunk_size == 128

    def test_project_file_overrides_user(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config import Settings
        saves = tmp_path / "saves"
        saves.mkdir()
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        user_file = tmp_path / "user.json"
        proj_file = tmp_path / "proj.json"
        self._write(user_file, {"embedding": {"model": "from-user"}, "chunk": {"size": 128}})
        self._write(proj_file, {"embedding": {"model": "from-project"}})
        monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(user_file))
        monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(proj_file))
        cfg = Settings()
        # project wins for embedding.model
        assert cfg.embedding_model == "from-project"
        # user still wins over default for chunk.size
        assert cfg.chunk_size == 128

    def test_env_overrides_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config import Settings
        saves = tmp_path / "saves"
        saves.mkdir()
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        proj_file = tmp_path / "proj.json"
        self._write(proj_file, {"embedding": {"model": "from-project"}})
        monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(tmp_path / "no_user.json"))
        monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(proj_file))
        monkeypatch.setenv("AGENT_KB_EMBEDDING_MODEL", "from-env")
        cfg = Settings()
        assert cfg.embedding_model == "from-env"


class TestLoadSettingsHelper:
    """load_settings() can substitute paths for a dry-run without env mutation."""

    def test_load_with_substituted_user_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config_files import load_settings
        saves = tmp_path / "saves"
        saves.mkdir()
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        # Ensure baseline discovery returns empty files
        monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(tmp_path / "base.json"))
        monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(tmp_path / "base_proj.json"))

        # Write a candidate file and pass it explicitly
        candidate = tmp_path / "candidate_user.json"
        candidate.write_text(json.dumps({"embedding": {"model": "candidate-model"}}))

        cfg = load_settings(user_path=candidate)
        assert cfg.embedding_model == "candidate-model"

    def test_load_resets_globals_on_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import agent_knowledgebase.config_files as cf
        from agent_knowledgebase.config_files import load_settings
        saves = tmp_path / "saves"
        saves.mkdir()
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(ValueError):
            load_settings(user_path=bad)
        assert cf._USER_CONFIG_PATH_OVERRIDE is None
        assert cf._PROJECT_CONFIG_PATH_OVERRIDE is None

    def test_load_with_no_args_uses_normal_discovery(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from agent_knowledgebase.config_files import load_settings
        saves = tmp_path / "saves"
        saves.mkdir()
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(tmp_path / "none.json"))
        monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(tmp_path / "none2.json"))
        cfg = load_settings()
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
