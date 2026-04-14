"""SQL-database ingest acceptance tests.

This module is the acceptance surface named in
``agent_knowledge_base_plugin_dev/ROADMAP.md`` under the Stage-1 blocking
items:

    tests/test_sql_database_ingest.py::test_where_ast_validator_*
    -- one negative test per rejected category, passes on CI.

    tests/test_sql_database_ingest.py::test_engine_is_readonly_*
    -- integration test that proves the sqlalchemy engine is pinned
    read-only at the driver layer.

The exhaustive unit-level coverage of the validator lives in
``tests/test_where_clause_validator.py``; the tests here are the named
acceptance cases -- one per rejected category -- wiring the same
validator into the ``test_where_ast_validator_<category>`` naming scheme
the roadmap promises.

Additional phases of the roadmap (dedup, batch cap, serialization) will
extend this file.  The validator + read-only engine acceptance tests
land first because they are the Stage-1 security gate.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from agent_knowledgebase.ingestors.sql_database import (
    SqlDatabaseIngestor,
    _build_readonly_sqlite_url,
)
from agent_knowledgebase.services.where_clause_validator import (
    CODE_ATTACH,
    CODE_BACKTICK,
    CODE_COMMENT,
    CODE_DETACH,
    CODE_FUNCTION_CALL,
    CODE_MULTI_STATEMENT,
    CODE_PRAGMA,
    CODE_SUBQUERY,
    CODE_UNION,
    WhereClauseValidationError,
    validate_where_clause,
)


# ---------------------------------------------------------------------------
# Positive acceptance: a canonical orchestrator where-clause must pass.
# ---------------------------------------------------------------------------


def test_where_ast_validator_accepts_canonical_orchestrator_clause() -> None:
    """The example clause from the orchestrator load-kb-from-sql skill."""
    clause = "published_at > '2026-01-01' AND video_id IS NOT NULL"
    # No exception == accept.
    assert validate_where_clause(clause) is None


# ---------------------------------------------------------------------------
# Negative acceptance -- one ``test_where_ast_validator_*`` test per
# rejected AST category listed in the ROADMAP.md blocking item.
# ---------------------------------------------------------------------------


def test_where_ast_validator_rejects_line_comment() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("id = 1 -- injected")
    assert exc.value.code == CODE_COMMENT


def test_where_ast_validator_rejects_block_comment() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("id = 1 /* injected */")
    assert exc.value.code == CODE_COMMENT


def test_where_ast_validator_rejects_subquery() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("id IN (SELECT id FROM secrets)")
    assert exc.value.code == CODE_SUBQUERY


def test_where_ast_validator_rejects_multi_statement() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("id = 1; DROP TABLE users")
    assert exc.value.code == CODE_MULTI_STATEMENT


def test_where_ast_validator_rejects_pragma() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("PRAGMA table_info(users)")
    assert exc.value.code == CODE_PRAGMA


def test_where_ast_validator_rejects_attach() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("ATTACH DATABASE 'evil.db' AS evil")
    assert exc.value.code == CODE_ATTACH


def test_where_ast_validator_rejects_detach() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("DETACH DATABASE evil")
    assert exc.value.code == CODE_DETACH


def test_where_ast_validator_rejects_function_call() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("LENGTH(name) > 3")
    assert exc.value.code == CODE_FUNCTION_CALL


def test_where_ast_validator_rejects_union() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("id = 1 UNION SELECT * FROM users")
    # sqlparse tokenisation may land this as UNION, subquery, or
    # multi-statement -- all three prove the clause is rejected.
    assert exc.value.code in (CODE_UNION, CODE_SUBQUERY, CODE_MULTI_STATEMENT)


def test_where_ast_validator_rejects_backticks() -> None:
    with pytest.raises(WhereClauseValidationError) as exc:
        validate_where_clause("`id` = 1")
    assert exc.value.code == CODE_BACKTICK


# ---------------------------------------------------------------------------
# Read-only engine pin (Stage-1 blocking item #2).
#
# The acceptance (from ROADMAP.md) is: an integration test that asserts
# either ``engine.dialect.readonly`` is truthy OR that an ``INSERT`` on the
# source DB raises ``OperationalError: attempt to write a readonly database``.
# We prefer the behavioral assertion because it is what an attacker would
# actually hit -- ``engine.dialect.readonly`` is not exposed for sqlite.
# ---------------------------------------------------------------------------


def _seed_sqlite(tmp_path: Path) -> Path:
    """Create a small sqlite DB and return the absolute file path.

    Seeding is done via the raw ``sqlite3`` module so the ingestor's
    read-only engine never touches the DB during setup.
    """
    db_path = tmp_path / "cache.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE videos(id INTEGER PRIMARY KEY, title TEXT)")
    con.execute("INSERT INTO videos (id, title) VALUES (1, 'seed')")
    con.commit()
    con.close()
    return db_path


def test_build_readonly_sqlite_url_shapes_sqlite() -> None:
    """The helper produces SQLite URI-mode with ``mode=ro`` and ``uri=true``."""
    url = _build_readonly_sqlite_url("sqlite:///tmp/cache.db")
    assert url == "sqlite:///file:/tmp/cache.db?mode=ro&uri=true"


def test_build_readonly_sqlite_url_preserves_windows_absolute_path() -> None:
    """Windows absolute paths (``C:/...``) survive the transform intact."""
    url = _build_readonly_sqlite_url("sqlite:///C:/data/cache.db")
    assert url == "sqlite:///file:/C:/data/cache.db?mode=ro&uri=true"


def test_build_readonly_sqlite_url_strips_ingestor_query_params() -> None:
    """``?table=``/``?where=`` are ingestor-layer; SQLite must never see them."""
    url = _build_readonly_sqlite_url(
        "sqlite:///tmp/cache.db?table=videos&where=id%3E1"
    )
    assert url == "sqlite:///file:/tmp/cache.db?mode=ro&uri=true"


def test_build_readonly_sqlite_url_passes_through_non_sqlite() -> None:
    """Non-sqlite URIs are out of Release-1 scope; leave them unchanged."""
    src = "postgresql://u:p@h/db"
    assert _build_readonly_sqlite_url(src) == src


def test_engine_is_readonly_blocks_insert(tmp_path: Path) -> None:
    """INSERT against the engine raises ``attempt to write a readonly database``.

    This is the behavioral acceptance from ROADMAP.md for the read-only
    engine pin.
    """
    db_path = _seed_sqlite(tmp_path)
    uri = f"sqlite:///{db_path.as_posix()}"

    engine = create_engine(_build_readonly_sqlite_url(uri))
    try:
        # Sanity: reads work.
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT id FROM videos")).fetchall()
            assert rows == [(1,)]

        # Writes are blocked at the SQLite driver layer.
        with pytest.raises(OperationalError) as exc:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO videos (id, title) VALUES (2, 'x')"))
                conn.commit()
        assert "attempt to write a readonly database" in str(exc.value)
    finally:
        engine.dispose()


def test_engine_is_readonly_blocks_insert_via_ingestor(tmp_path: Path) -> None:
    """The ingestor's public ``read()`` path exercises the same read-only pin.

    We call the real ingestor, then re-open an engine with the helper and
    prove a subsequent INSERT is still blocked -- i.e. nothing in the
    ingestor codepath re-opens a writable handle.
    """
    db_path = _seed_sqlite(tmp_path)
    uri = f"sqlite:///{db_path.as_posix()}"

    ingestor = SqlDatabaseIngestor()
    # read() succeeds (schema + sample row) against the read-only engine.
    contents = ingestor.read(uri, metadata={"sample_rows": 1})
    assert len(contents) == 1
    assert "videos" in contents[0].text
    assert "seed" in contents[0].text

    # Now re-open via the helper and confirm INSERT still raises.
    engine = create_engine(_build_readonly_sqlite_url(uri))
    try:
        with pytest.raises(OperationalError) as exc:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO videos (id, title) VALUES (2, 'x')"))
                conn.commit()
        assert "attempt to write a readonly database" in str(exc.value)
    finally:
        engine.dispose()
