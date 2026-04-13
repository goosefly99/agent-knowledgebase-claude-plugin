"""Tests for config file path resolution and JSON source."""

from __future__ import annotations

import os
from pathlib import Path

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
