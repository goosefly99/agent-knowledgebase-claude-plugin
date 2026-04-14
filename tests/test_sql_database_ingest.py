"""SQL-database ingest acceptance tests.

This module is the acceptance surface named in
``agent_knowledge_base_plugin_dev/ROADMAP.md`` under the first blocking
item:

    tests/test_sql_database_ingest.py::test_where_ast_validator_*
    -- one negative test per rejected category, passes on CI.

The exhaustive unit-level coverage of the validator lives in
``tests/test_where_clause_validator.py``; the tests here are the named
acceptance cases -- one per rejected category -- wiring the same
validator into the ``test_where_ast_validator_<category>`` naming scheme
the roadmap promises.

Additional phases of the roadmap (read-only engine, dedup, batch cap,
serialization) will extend this file.  The validator acceptance tests
land first because the SQL-injection AST validator is the security gate
that blocks the entire Stage 1 release.
"""

from __future__ import annotations

import pytest

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
