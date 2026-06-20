"""Shared test fixtures for agent-knowledgebase."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.database import Database


@pytest.fixture(autouse=True)
def _isolate_from_user_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Point config path discovery at nonexistent per-session tmp files.

    Without this, tests that construct :class:`Settings` without explicit
    path overrides would pick up the developer's real
    ``~/.agent-kb/config.json`` and fail non-deterministically based on
    host state. Individual tests that want to exercise real path
    discovery override these env vars themselves.
    """
    empty_dir = tmp_path_factory.mktemp("isolated_config_roots")
    monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(empty_dir / "no_user.json"))
    monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(empty_dir / "no_project.json"))


@pytest.fixture()
def test_config(tmp_path: Path) -> Settings:
    """Return a :class:`Settings` with *saves_dir* pointing to ``tmp_path``.

    The directory is pre-created so that ``resolve_paths()`` will not raise.
    """
    saves = tmp_path / "saves"
    saves.mkdir()
    return Settings(
        saves_dir=saves,
        export_path=tmp_path / "export",
    )


@pytest.fixture()
def test_db(tmp_path: Path) -> Database:
    """Return a :class:`Database` backed by a temporary SQLite file."""
    db = Database(db_path=tmp_path / "test.db")
    yield db  # type: ignore[misc]
    db.close()
