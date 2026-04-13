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
