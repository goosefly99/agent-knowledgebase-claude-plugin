"""Shared test fixtures for agent-knowledgebase."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.database import Database


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
