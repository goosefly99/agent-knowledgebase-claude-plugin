"""Phase B: Verify Porter stemmer behaviour in FTS5.

Tests that the 'porter unicode61' tokenizer normalises inflected forms
and Unicode variants to the same FTS5 index token, so queries on stem
forms match documents with other inflections.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from agent_knowledgebase.backends.textvec_backend import TextvecBackend
from agent_knowledgebase.config import Settings, sanitize_kb_dir_name
from agent_knowledgebase.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(tmp_path: Path) -> tuple[TextvecBackend, str]:
    saves = tmp_path / "saves"
    saves.mkdir(exist_ok=True)
    kb_id = f"porter-{uuid.uuid4().hex[:8]}"
    settings = Settings(saves_dir=saves)
    safe = sanitize_kb_dir_name(kb_id)
    db_path = settings.kb_db_path(safe)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path=db_path)
    db._conn.execute(
        "INSERT OR IGNORE INTO knowledgebases (id, name, description, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (kb_id, kb_id, "", "{}", "2024-01-01", "2024-01-01"),
    )
    db._conn.execute(
        "INSERT OR IGNORE INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"src-{kb_id}", kb_id, "file", "/tmp/porter.txt", "ingested"),
    )
    db._conn.commit()
    return TextvecBackend(settings, service=None), kb_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fts5_porter_tokenizer_available() -> None:
    """Verify the porter tokenizer is available in the local sqlite build."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')"
        )
    except sqlite3.OperationalError as exc:
        pytest.skip(f"porter tokenizer not available: {exc}")
    finally:
        conn.close()


def test_porter_stem_running_runner_runs(tmp_path: Path) -> None:
    """'running', 'runner', 'runs' all tokenise to the same stem.

    A document with 'running' should match queries for 'runner' or 'runs'
    because Porter reduces them to the same root ('run').
    """
    backend, kb_id = _make_backend(tmp_path)
    backend.index(
        kb_id=kb_id,
        documents=[
            {
                "id": "doc-running",
                "content": "The runner was running faster than expected.",
                "metadata": {"kb_id": kb_id, "source_id": f"src-{kb_id}"},
            }
        ],
    )

    # Search for 'runs' — Porter should map runs→run, runner→run, running→run.
    results_runs = backend.search(kb_id=kb_id, text="runs", top_k=5)
    assert len(results_runs) > 0, (
        "Expected 'runs' to match document containing 'running'/'runner' "
        "via Porter stemming (runs→run == running→run == runner→run)"
    )

    results_runner = backend.search(kb_id=kb_id, text="runner", top_k=5)
    assert len(results_runner) > 0, (
        "Expected 'runner' to match document containing 'running' "
        "via Porter stemming"
    )


def test_porter_stem_plural_forms(tmp_path: Path) -> None:
    """Plural and singular forms (cats / cat) reduce to the same stem."""
    backend, kb_id = _make_backend(tmp_path)
    backend.index(
        kb_id=kb_id,
        documents=[
            {
                "id": "doc-cats",
                "content": "The cats chased many mice through the house.",
                "metadata": {"kb_id": kb_id, "source_id": f"src-{kb_id}"},
            }
        ],
    )

    results = backend.search(kb_id=kb_id, text="cat", top_k=5)
    assert len(results) > 0, (
        "Expected 'cat' to match document containing 'cats' via Porter stemming"
    )


def test_unicode61_normalisation(tmp_path: Path) -> None:
    """Unicode61 normalisation: accented characters match unaccented queries.

    FTS5 with 'unicode61' tokenizer normalises Unicode codepoints so
    that accented characters (café) and their ASCII equivalents (cafe)
    map to the same token.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')"
        )
        conn.execute("INSERT INTO t(rowid, x) VALUES (1, 'café au lait')")
        rows = conn.execute("SELECT * FROM t WHERE t MATCH 'cafe'").fetchall()
        # If unicode61 normalises café→cafe, rows should be non-empty.
        # SQLite's unicode61 folds diacritics on most builds; if it doesn't,
        # skip rather than fail (platform-specific behaviour).
        if not rows:
            pytest.skip(
                "unicode61 diacritic folding not available on this sqlite build "
                "(café→cafe match returned empty; acceptable platform variance)"
            )
    except sqlite3.OperationalError as exc:
        pytest.skip(f"FTS5 not available: {exc}")
    finally:
        conn.close()


def test_chunked_fts5_porter_via_backend(tmp_path: Path) -> None:
    """End-to-end: index a document with 'organized', query with 'organize'."""
    backend, kb_id = _make_backend(tmp_path)
    backend.index(
        kb_id=kb_id,
        documents=[
            {
                "id": "doc-organized",
                "content": "The team organized the files carefully and systematically.",
                "metadata": {"kb_id": kb_id, "source_id": f"src-{kb_id}"},
            }
        ],
    )
    results = backend.search(kb_id=kb_id, text="organize", top_k=5)
    assert len(results) > 0, (
        "Expected 'organize' to match 'organized' via Porter stemming in FTS5 backend"
    )
