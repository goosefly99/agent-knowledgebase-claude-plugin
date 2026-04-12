"""Shared test fixtures for agent-knowledgebase."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.database import Database


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a temporary SQLite database path inside ``tmp_path``."""
    return tmp_path / "test_knowledgebase.db"


@pytest.fixture()
def test_config(tmp_path: Path, tmp_db_path: Path) -> Settings:
    """Return a :class:`Settings` instance with paths pointing to ``tmp_path``."""
    return Settings(
        db_path=tmp_db_path,
        chroma_path=tmp_path / "chroma",
        export_path=tmp_path / "export",
    )


@pytest.fixture()
def test_db(tmp_path: Path) -> Database:
    """Return a :class:`Database` backed by a temporary SQLite file."""
    db = Database(db_path=tmp_path / "test.db")
    yield db  # type: ignore[misc]
    db.close()
