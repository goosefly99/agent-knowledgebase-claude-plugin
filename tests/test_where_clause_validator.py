"""Unit tests for :mod:`agent_knowledgebase.services.where_clause_validator`.

Each negative test targets one rejected AST category from the release spec:
comments, subqueries, multi-statements, PRAGMA, ATTACH, DETACH, function
calls, UNION, backticks.  Positive tests cover the allowed constructs
(comparison ops, logicals, IN, LIKE, IS NULL).
"""

from __future__ import annotations

import pytest

from agent_knowledgebase.services.where_clause_validator import (
    CODE_ARITHMETIC,
    CODE_ATTACH,
    CODE_BACKTICK,
    CODE_COMMENT,
    CODE_DETACH,
    CODE_DISALLOWED_KEYWORD,
    CODE_EMPTY,
    CODE_FUNCTION_CALL,
    CODE_MULTI_STATEMENT,
    CODE_PRAGMA,
    CODE_SUBQUERY,
    CODE_UNION,
    WhereClauseValidationError,
    validate_where_clause,
)


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


class TestAccepted:
    """Allowed where-clauses must pass silently."""

    @pytest.mark.parametrize(
        "clause",
        [
            "published_at > '2026-01-01'",
            "id = 42",
            "id != 42",
            "id <> 42",
            "views >= 100 AND views <= 1000",
            "status IN ('ok', 'pending')",
            "status NOT IN ('error', 'cancelled')",
            "title LIKE 'intro%'",
            "title NOT LIKE '%draft%'",
            "deleted_at IS NULL",
            "deleted_at IS NOT NULL",
            "(a = 1 OR a = 2) AND b = 3",
            "NOT (a = 1)",
            'col_with_quotes = "value"',
            "col = 'it''s fine'",  # doubled-quote literal escape
            "col = 'a;b'",  # semicolon inside literal is fine
        ],
    )
    def test_allows_valid_where(self, clause: str) -> None:
        assert validate_where_clause(clause) is None

    def test_tolerates_leading_where_keyword(self) -> None:
        assert validate_where_clause("WHERE id = 1") is None


# ---------------------------------------------------------------------------
# Negative cases -- one per rejected AST category
# ---------------------------------------------------------------------------


class TestRejectsComments:
    def test_line_comment(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id = 1 -- drop the bomb")
        assert exc.value.code == CODE_COMMENT

    def test_block_comment(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id = 1 /* still bad */")
        assert exc.value.code == CODE_COMMENT


class TestRejectsSubqueries:
    def test_in_subquery(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id IN (SELECT id FROM secrets)")
        assert exc.value.code == CODE_SUBQUERY

    def test_bare_select_in_paren(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id = (SELECT MAX(id) FROM users)")
        assert exc.value.code == CODE_SUBQUERY


class TestRejectsMultiStatement:
    def test_trailing_semicolon_and_statement(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id = 1; DROP TABLE users")
        assert exc.value.code == CODE_MULTI_STATEMENT

    def test_lone_trailing_semicolon(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id = 1;")
        assert exc.value.code == CODE_MULTI_STATEMENT


class TestRejectsPragma:
    def test_pragma(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("PRAGMA table_info(users)")
        assert exc.value.code == CODE_PRAGMA


class TestRejectsAttach:
    def test_attach(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("ATTACH DATABASE 'evil.db' AS evil")
        assert exc.value.code == CODE_ATTACH


class TestRejectsDetach:
    def test_detach(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("DETACH DATABASE evil")
        assert exc.value.code == CODE_DETACH


class TestRejectsFunctionCalls:
    def test_simple_function(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("LENGTH(name) > 3")
        assert exc.value.code == CODE_FUNCTION_CALL

    def test_nested_function(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id = ABS(RANDOM())")
        assert exc.value.code == CODE_FUNCTION_CALL


class TestRejectsUnion:
    def test_union(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id = 1 UNION SELECT * FROM users")
        # UNION lands first either as the UNION keyword or as a subquery
        # depending on sqlparse's tokenization.  Both codes are valid
        # rejections; assert the set.
        assert exc.value.code in (CODE_UNION, CODE_SUBQUERY, CODE_MULTI_STATEMENT)


class TestRejectsBackticks:
    def test_backtick_identifier(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("`id` = 1")
        assert exc.value.code == CODE_BACKTICK


# ---------------------------------------------------------------------------
# Misc boundary cases
# ---------------------------------------------------------------------------


class TestMiscRejections:
    def test_empty_string(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("")
        assert exc.value.code == CODE_EMPTY

    def test_none(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause(None)  # type: ignore[arg-type]
        assert exc.value.code == CODE_EMPTY

    def test_whitespace_only(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("   \t\n  ")
        assert exc.value.code == CODE_EMPTY

    def test_arithmetic_operator_rejected(self) -> None:
        # ``WHERE 1+1 = 2`` smuggles arithmetic -- reject.
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("1+1 = 2")
        assert exc.value.code == CODE_ARITHMETIC

    def test_bitwise_operator_rejected(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("flags & 1 = 1")
        assert exc.value.code == CODE_ARITHMETIC

    def test_disallowed_keyword_between(self) -> None:
        # BETWEEN is not in the allow-list per the spec.
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id BETWEEN 1 AND 10")
        assert exc.value.code == CODE_DISALLOWED_KEYWORD

    def test_insert_keyword_rejected(self) -> None:
        with pytest.raises(WhereClauseValidationError) as exc:
            validate_where_clause("id = 1 INSERT INTO x VALUES (1)")
        # Either MULTI_STATEMENT (if sqlparse treats it as separate) or
        # DISALLOWED_KEYWORD.  Both prove the clause was rejected.
        assert exc.value.code in (CODE_DISALLOWED_KEYWORD, CODE_MULTI_STATEMENT)
