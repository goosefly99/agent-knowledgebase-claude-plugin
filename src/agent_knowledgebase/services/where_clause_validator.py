"""sqlparse-AST allow-list validator for kb_ingest sql_database where-clauses.

This module is the security gate for ``source_type='sql_database'`` ingest.
Per release spec `spec-agent-knowledgebase-update-2026-04-13` and the
round-3 debate critic, we reject untrusted SQL by combining:

1. A pre-parse string-level scan that is literal-aware (tolerates quoted
   semicolons and the like) and rejects the classic injection shapes
   that sqlparse is not reliable at catching -- comments, multi-statement
   terminators, backticks, and the control-plane keywords
   ``UNION`` / ``INTERSECT`` / ``EXCEPT`` / ``PRAGMA`` / ``ATTACH`` /
   ``DETACH``.
2. An AST walk of the caller's where-clause (parsed inside a synthetic
   ``SELECT 1 FROM _t WHERE <expr>`` so sqlparse produces a proper
   :class:`sqlparse.sql.Where` node).  The walker rejects anything not
   on the allow-list: comparison ops, logical ops, identifiers, literals,
   parentheses-for-grouping, ``IN`` / ``NOT IN`` / ``LIKE`` / ``NOT LIKE``
   / ``IS NULL`` / ``IS NOT NULL``.

All rejections raise :class:`WhereClauseValidationError` with a stable
``code`` and a human readable ``message``.  Callers surface the
structured error to the MCP tool boundary.
"""

from __future__ import annotations

import re
from typing import Iterable

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import (
    Comparison,
    Function,
    Identifier,
    IdentifierList,
    Operation,
    Parenthesis,
    Statement,
    Token,
    TokenList,
    Where,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Error codes -- stable; the MCP layer embeds these in structured responses.
CODE_EMPTY = "WHERE_EMPTY"
CODE_COMMENT = "WHERE_CONTAINS_COMMENT"
CODE_MULTI_STATEMENT = "WHERE_MULTI_STATEMENT"
CODE_SUBQUERY = "WHERE_SUBQUERY_FORBIDDEN"
CODE_FUNCTION_CALL = "WHERE_FUNCTION_CALL_FORBIDDEN"
CODE_UNION = "WHERE_SET_OPERATOR_FORBIDDEN"
CODE_PRAGMA = "WHERE_PRAGMA_FORBIDDEN"
CODE_ATTACH = "WHERE_ATTACH_FORBIDDEN"
CODE_DETACH = "WHERE_DETACH_FORBIDDEN"
CODE_BACKTICK = "WHERE_BACKTICK_FORBIDDEN"
CODE_ARITHMETIC = "WHERE_ARITHMETIC_FORBIDDEN"
CODE_DISALLOWED_KEYWORD = "WHERE_DISALLOWED_KEYWORD"
CODE_DISALLOWED_OPERATOR = "WHERE_DISALLOWED_OPERATOR"
CODE_PARSE_FAILED = "WHERE_PARSE_FAILED"


class WhereClauseValidationError(ValueError):
    """Structured error raised when a where-clause fails the allow-list.

    Attributes:
        code: Stable machine-readable error code (see ``CODE_*`` constants).
        message: Human-readable remediation hint.
        offending_token: The textual slice that triggered the rejection,
            if a single token can be blamed; empty string otherwise.
    """

    def __init__(self, code: str, message: str, offending_token: str = "") -> None:
        self.code = code
        self.message = message
        self.offending_token = offending_token
        suffix = f" (token={offending_token!r})" if offending_token else ""
        super().__init__(f"[{code}] {message}{suffix}")


# ---------------------------------------------------------------------------
# Allow-lists / deny-lists
# ---------------------------------------------------------------------------

_ALLOWED_COMPARISON_OPERATORS: frozenset[str] = frozenset(
    {"=", "!=", "<>", "<", "<=", ">", ">="}
)

# Comparison-family keywords that sqlparse emits as ``T.Operator.Comparison``
# (e.g. ``LIKE``, ``NOT LIKE``, ``IN``, ``NOT IN``).
_ALLOWED_COMPARISON_KEYWORDS: frozenset[str] = frozenset(
    {"LIKE", "NOT LIKE", "IN", "NOT IN"}
)

# Logical keywords plus the null-predicate keywords.  Composite forms
# that sqlparse emits as a single token are listed explicitly
# (``NOT NULL``, ``NOT IN``, ``IS NOT``).
_ALLOWED_KEYWORDS: frozenset[str] = frozenset(
    {
        "AND",
        "OR",
        "NOT",
        "IN",
        "LIKE",
        "IS",
        "NULL",
        "NOT NULL",
        "IS NOT",
        "IS NOT NULL",
        "NOT IN",
        "NOT LIKE",
        "ESCAPE",  # ``LIKE '%x' ESCAPE '\\'`` is benign
    }
)

# Keywords that MUST NEVER reach the AST walk -- they are rejected at
# the string-level guard before parsing.  Each maps to the dedicated
# error code the callers assert on.
_STRING_LEVEL_FORBIDDEN_KEYWORDS: dict[str, str] = {
    "PRAGMA": CODE_PRAGMA,
    "ATTACH": CODE_ATTACH,
    "DETACH": CODE_DETACH,
    "UNION": CODE_UNION,
    "INTERSECT": CODE_UNION,
    "EXCEPT": CODE_UNION,
}

# Keywords we reject inside the AST walk (defense-in-depth; the string
# guard usually catches them first).  These are all DML/DDL keywords
# that have no business in a where-clause.
_AST_LEVEL_FORBIDDEN_KEYWORDS: dict[str, str] = {
    "SELECT": CODE_SUBQUERY,
    "INSERT": CODE_DISALLOWED_KEYWORD,
    "UPDATE": CODE_DISALLOWED_KEYWORD,
    "DELETE": CODE_DISALLOWED_KEYWORD,
    "DROP": CODE_DISALLOWED_KEYWORD,
    "CREATE": CODE_DISALLOWED_KEYWORD,
    "ALTER": CODE_DISALLOWED_KEYWORD,
    "TRUNCATE": CODE_DISALLOWED_KEYWORD,
    "GRANT": CODE_DISALLOWED_KEYWORD,
    "REVOKE": CODE_DISALLOWED_KEYWORD,
    "EXEC": CODE_DISALLOWED_KEYWORD,
    "EXECUTE": CODE_DISALLOWED_KEYWORD,
    "VACUUM": CODE_DISALLOWED_KEYWORD,
    "REINDEX": CODE_DISALLOWED_KEYWORD,
    "INTO": CODE_DISALLOWED_KEYWORD,
    "VALUES": CODE_DISALLOWED_KEYWORD,
    "MERGE": CODE_DISALLOWED_KEYWORD,
    "CALL": CODE_DISALLOWED_KEYWORD,
    # set operators -- belt & suspenders; string guard also catches.
    "UNION": CODE_UNION,
    "INTERSECT": CODE_UNION,
    "EXCEPT": CODE_UNION,
    # between / case / when -- not in the allow-list per spec
    "BETWEEN": CODE_DISALLOWED_KEYWORD,
    "CASE": CODE_DISALLOWED_KEYWORD,
    "WHEN": CODE_DISALLOWED_KEYWORD,
    "THEN": CODE_DISALLOWED_KEYWORD,
    "ELSE": CODE_DISALLOWED_KEYWORD,
    "END": CODE_DISALLOWED_KEYWORD,
    "JOIN": CODE_DISALLOWED_KEYWORD,
    "ON": CODE_DISALLOWED_KEYWORD,
    "GROUP": CODE_DISALLOWED_KEYWORD,
    "BY": CODE_DISALLOWED_KEYWORD,
    "ORDER": CODE_DISALLOWED_KEYWORD,
    "HAVING": CODE_DISALLOWED_KEYWORD,
    "LIMIT": CODE_DISALLOWED_KEYWORD,
    "OFFSET": CODE_DISALLOWED_KEYWORD,
    "CAST": CODE_DISALLOWED_KEYWORD,
    # windowing / CTE
    "WITH": CODE_DISALLOWED_KEYWORD,
    "RECURSIVE": CODE_DISALLOWED_KEYWORD,
    "OVER": CODE_DISALLOWED_KEYWORD,
}

# Pre-compiled word-boundary regex used by the string-level keyword
# guard.  We use ``\b`` so ``UNIONIZED`` as a column name would not
# match -- though such a column would be bizarre in a cache DB.
_FORBIDDEN_KEYWORD_PATTERNS: dict[str, "re.Pattern[str]"] = {
    kw: re.compile(rf"\b{kw}\b", re.IGNORECASE)
    for kw in _STRING_LEVEL_FORBIDDEN_KEYWORDS
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate_where_clause(where: str) -> None:
    """Validate a caller-provided ``where=`` expression against the allow-list.

    This function either returns ``None`` (accept) or raises
    :class:`WhereClauseValidationError`.  It never returns a value on
    rejection -- callers must treat the absence of an exception as success.

    The parser is :mod:`sqlparse`.  We prepend ``SELECT 1 FROM _t WHERE ``
    so sqlparse produces a proper :class:`sqlparse.sql.Where` node with
    full structure; this avoids the ambiguity of parsing a bare expression.

    Args:
        where: The raw where-clause string exactly as the caller passed it.
               Leading ``WHERE`` prefix is tolerated and stripped.

    Raises:
        WhereClauseValidationError: If any rejected construct is present.
    """
    if where is None or not str(where).strip():
        raise WhereClauseValidationError(
            code=CODE_EMPTY,
            message="where-clause must be a non-empty string",
        )

    raw = str(where).strip()

    # --- Pre-parse string-level guards (literal-aware) -------------------
    _reject_backticks(raw)
    _reject_raw_comments(raw)
    non_literal = _strip_literals_and_comments(raw)
    _reject_multi_statement(non_literal)
    _reject_forbidden_keywords(non_literal)

    # Tolerate a leading ``WHERE`` keyword -- the tool's uri grammar strips
    # it but be defensive.
    stripped = raw[6:].lstrip() if raw[:6].upper() == "WHERE " else raw

    wrapped = f"SELECT 1 FROM _t WHERE {stripped}"
    try:
        parsed = sqlparse.parse(wrapped)
    except Exception as exc:  # pragma: no cover - sqlparse is forgiving
        raise WhereClauseValidationError(
            code=CODE_PARSE_FAILED,
            message=f"sqlparse failed to parse where-clause: {exc}",
        ) from exc

    if len(parsed) == 0:
        raise WhereClauseValidationError(
            code=CODE_PARSE_FAILED,
            message="sqlparse returned no statements for the wrapped query",
        )
    if len(parsed) > 1:
        raise WhereClauseValidationError(
            code=CODE_MULTI_STATEMENT,
            message="where-clause must contain exactly one statement",
        )

    stmt: Statement = parsed[0]

    # --- Top-level sibling check -----------------------------------------
    # sqlparse puts UNION and trailing statements at the Statement level,
    # not inside the Where node.  Validate that only the expected
    # ``SELECT 1 FROM _t`` + Where + trailing whitespace exists.
    _validate_statement_shape(stmt)

    where_node = _find_where(stmt)
    if where_node is None:
        raise WhereClauseValidationError(
            code=CODE_PARSE_FAILED,
            message="where-clause did not produce a WHERE node after wrapping",
        )

    _walk(where_node)


# ---------------------------------------------------------------------------
# String-level guards
# ---------------------------------------------------------------------------


def _reject_backticks(raw: str) -> None:
    if "`" in raw:
        raise WhereClauseValidationError(
            code=CODE_BACKTICK,
            message="backtick-quoted identifiers are not allowed; use double quotes",
            offending_token="`",
        )


def _reject_raw_comments(raw: str) -> None:
    """Reject comments at the string level."""
    if "--" in raw:
        raise WhereClauseValidationError(
            code=CODE_COMMENT,
            message="SQL comments are not allowed in where-clauses",
            offending_token="--",
        )
    if "/*" in raw or "*/" in raw:
        raise WhereClauseValidationError(
            code=CODE_COMMENT,
            message="SQL block comments are not allowed in where-clauses",
            offending_token="/*",
        )


def _strip_literals_and_comments(raw: str) -> str:
    """Return a copy of ``raw`` with string literals replaced by placeholder ``x``.

    This is used by later string-level checks so a literal like
    ``'UNION'`` doesn't trigger a false-positive UNION rejection.
    Comments have already been rejected outright by
    :func:`_reject_raw_comments`, so they cannot appear here.
    """
    out: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(raw) and raw[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
            out.append("x" if in_single else "x")
            i += 1
            continue
        if ch == '"' and not in_single:
            if in_double and i + 1 < len(raw) and raw[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
            # keep the double-quote so identifier parsing still works
            out.append(ch)
            i += 1
            continue
        if in_single:
            # replace literal body with placeholder; don't echo original
            out.append("x")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _reject_multi_statement(non_literal: str) -> None:
    """Reject any ``;`` in the non-literal portion of the input."""
    if ";" in non_literal:
        raise WhereClauseValidationError(
            code=CODE_MULTI_STATEMENT,
            message="semicolons (statement terminators) are not allowed",
            offending_token=";",
        )


def _reject_forbidden_keywords(non_literal: str) -> None:
    """Reject PRAGMA / ATTACH / DETACH / UNION / INTERSECT / EXCEPT.

    This is the authoritative check -- callers assert on specific
    :data:`CODE_*` values (e.g. ``CODE_PRAGMA``), so the order here
    determines which code wins when an input contains multiple forbidden
    keywords.  We therefore iterate in spec-order (PRAGMA, ATTACH,
    DETACH, UNION, INTERSECT, EXCEPT) rather than dict-insertion order.
    """
    for kw in ("PRAGMA", "ATTACH", "DETACH", "UNION", "INTERSECT", "EXCEPT"):
        pat = _FORBIDDEN_KEYWORD_PATTERNS[kw]
        m = pat.search(non_literal)
        if m:
            raise WhereClauseValidationError(
                code=_STRING_LEVEL_FORBIDDEN_KEYWORDS[kw],
                message=f"keyword {kw!r} is not allowed in a where-clause",
                offending_token=m.group(0),
            )


# ---------------------------------------------------------------------------
# Statement-shape validation (top-level)
# ---------------------------------------------------------------------------


def _validate_statement_shape(stmt: Statement) -> None:
    """Reject anything at the Statement level that isn't ``SELECT 1 FROM _t [Where] [ws]``.

    The string guard already rejected UNION etc., but in case sqlparse
    ever evolves we enforce the expected siblings here too.
    """
    allowed_top_level_types = (Where,)
    # Walk the Statement's direct tokens and ensure only: SELECT, 1, FROM,
    # _t Identifier, whitespace, Where, and no other structural tokens.
    for tok in stmt.tokens:
        if tok.is_whitespace:
            continue
        if isinstance(tok, allowed_top_level_types):
            continue
        if isinstance(tok, Identifier) and tok.value == "_t":
            continue
        if tok.ttype is T.Keyword.DML and tok.normalized.upper() == "SELECT":
            continue
        if tok.ttype is T.Keyword and tok.normalized.upper() == "FROM":
            continue
        if tok.ttype is T.Literal.Number.Integer and tok.value == "1":
            continue
        if tok.ttype is T.Punctuation:
            continue
        # Anything else at the top level means the caller smuggled
        # something past the string guard.
        raise WhereClauseValidationError(
            code=CODE_MULTI_STATEMENT,
            message="unexpected top-level token outside the where-clause",
            offending_token=tok.value,
        )


def _find_where(stmt: Statement) -> Where | None:
    """Locate the single Where node inside a wrapped SELECT statement."""
    for tok in stmt.tokens:
        if isinstance(tok, Where):
            return tok
    return None


# ---------------------------------------------------------------------------
# AST walk
# ---------------------------------------------------------------------------


def _walk(node: TokenList) -> None:
    """Recursively validate every token in ``node`` against the allow-list."""
    for tok in node.tokens:
        if _is_ignorable(tok):
            continue
        _check_token(tok)
        if isinstance(tok, TokenList):
            # Don't recurse into constructs whose children we've already
            # validated structurally (Function already raised; Parenthesis
            # bounds were checked; Operation already raised).
            if isinstance(tok, (Function, Operation)):
                continue
            _walk(tok)


def _is_ignorable(tok: Token) -> bool:
    """Whitespace / top-level Where-keyword / benign punctuation."""
    if tok.is_whitespace:
        return True
    if tok.ttype in (T.Whitespace, T.Newline):
        return True
    if tok.ttype is T.Keyword and tok.normalized.upper() == "WHERE":
        return True
    if tok.ttype is T.Punctuation and tok.value != ";":
        return True
    return False


def _check_token(tok: Token) -> None:
    """Reject token ``tok`` if it is outside the allow-list."""
    # 1. Comments
    if tok.ttype is not None and str(tok.ttype).startswith("Token.Comment"):
        raise WhereClauseValidationError(
            code=CODE_COMMENT,
            message="SQL comments are not allowed in where-clauses",
            offending_token=tok.value,
        )

    # 2. Arithmetic / bitwise operations -- sqlparse wraps these in an
    # ``Operation`` TokenList node.  Reject unconditionally.
    if isinstance(tok, Operation):
        raise WhereClauseValidationError(
            code=CODE_ARITHMETIC,
            message="arithmetic or bitwise operations are not allowed",
            offending_token=tok.value,
        )

    # 3. Subqueries: Parenthesis containing SELECT.
    if isinstance(tok, Parenthesis):
        if _contains_select(tok):
            raise WhereClauseValidationError(
                code=CODE_SUBQUERY,
                message="subqueries inside where-clauses are forbidden",
                offending_token=tok.value,
            )
        return  # grouping paren: _walk will recurse

    # 4. Function calls.
    if isinstance(tok, Function):
        raise WhereClauseValidationError(
            code=CODE_FUNCTION_CALL,
            message="function calls are not allowed in where-clauses",
            offending_token=tok.value,
        )

    # 5. Structural TokenLists -- recursion will validate children.
    if isinstance(tok, (Comparison, Identifier, IdentifierList)):
        return

    # 6. Keyword-level checks.  sqlparse over-tags plain identifiers as
    # Keyword (e.g. ``views`` becomes T.Keyword), so the strategy is:
    # - DML / DDL / CTE keywords: must be on the AST allow-list or
    #   explicitly forbidden.
    # - Plain T.Keyword tokens: allowed if in ``_ALLOWED_KEYWORDS`` or
    #   in the AST denylist; otherwise treat as identifier (sqlparse
    #   false-positive).
    if tok.ttype in (T.Keyword.DML, T.Keyword.DDL, T.Keyword.CTE):
        upper = tok.normalized.upper()
        if upper in _AST_LEVEL_FORBIDDEN_KEYWORDS:
            raise WhereClauseValidationError(
                code=_AST_LEVEL_FORBIDDEN_KEYWORDS[upper],
                message=f"keyword {upper!r} is not allowed in a where-clause",
                offending_token=tok.value,
            )
        # DML/DDL/CTE not in allow-list and not explicitly forbidden ->
        # reject to be safe.
        if upper not in _ALLOWED_KEYWORDS:
            raise WhereClauseValidationError(
                code=CODE_DISALLOWED_KEYWORD,
                message=f"keyword {upper!r} is not in the allow-list",
                offending_token=tok.value,
            )
        return

    if tok.ttype is T.Keyword:
        upper = tok.normalized.upper()
        if upper in _AST_LEVEL_FORBIDDEN_KEYWORDS:
            raise WhereClauseValidationError(
                code=_AST_LEVEL_FORBIDDEN_KEYWORDS[upper],
                message=f"keyword {upper!r} is not allowed in a where-clause",
                offending_token=tok.value,
            )
        if upper in _ALLOWED_KEYWORDS:
            return
        # sqlparse false-positives common column names as keywords.
        # Accept any single-word uppercase-letter token that isn't
        # explicitly forbidden and doesn't contain whitespace.  Whitespace
        # indicates a composite keyword like ``NOT NULL`` that we need
        # to explicitly allow or deny.
        if " " in upper or "\t" in upper or "\n" in upper:
            raise WhereClauseValidationError(
                code=CODE_DISALLOWED_KEYWORD,
                message=f"composite keyword {upper!r} is not allowed",
                offending_token=tok.value,
            )
        # Treat as identifier.
        return

    # 7. Operators.
    if tok.ttype is T.Operator.Comparison:
        if tok.value in _ALLOWED_COMPARISON_OPERATORS:
            return
        upper = tok.value.upper()
        if upper in _ALLOWED_COMPARISON_KEYWORDS:
            return
        raise WhereClauseValidationError(
            code=CODE_DISALLOWED_OPERATOR,
            message=f"comparison operator {tok.value!r} is not allowed",
            offending_token=tok.value,
        )
    if tok.ttype is T.Operator:
        # Bare ``T.Operator`` -> arithmetic / bitwise / concat.  These
        # are usually caught at the Operation-TokenList level above, but
        # be defensive.
        raise WhereClauseValidationError(
            code=CODE_ARITHMETIC,
            message=f"operator {tok.value!r} is not allowed",
            offending_token=tok.value,
        )

    # 8. Wildcards.
    if tok.ttype is T.Wildcard:
        raise WhereClauseValidationError(
            code=CODE_DISALLOWED_OPERATOR,
            message="wildcards (*) are not allowed in where-clauses",
            offending_token=tok.value,
        )

    # 9. Bind parameters.
    if tok.ttype in (T.Name.Placeholder,):
        raise WhereClauseValidationError(
            code=CODE_DISALLOWED_KEYWORD,
            message="bind parameters are not allowed in where-clauses",
            offending_token=tok.value,
        )

    # 10. Literals / identifiers.
    if tok.ttype is None:
        raise WhereClauseValidationError(
            code=CODE_PARSE_FAILED,
            message=f"unclassifiable token: {tok.value!r}",
            offending_token=tok.value,
        )

    allowed_prefixes = (
        "Token.Literal",
        "Token.Name",
        "Token.Number",
        "Token.String",
    )
    ttype_str = str(tok.ttype)
    if any(ttype_str.startswith(p) for p in allowed_prefixes):
        return

    raise WhereClauseValidationError(
        code=CODE_DISALLOWED_KEYWORD,
        message=f"token of type {ttype_str} is not in the allow-list",
        offending_token=tok.value,
    )


def _contains_select(node: TokenList) -> bool:
    """True if any descendant token is the ``SELECT`` keyword (DML)."""
    for tok in _iter_all_tokens(node):
        if (
            tok.ttype in (T.Keyword.DML, T.Keyword)
            and tok.normalized.upper() == "SELECT"
        ):
            return True
    return False


def _iter_all_tokens(node: TokenList) -> Iterable[Token]:
    for tok in node.tokens:
        yield tok
        if isinstance(tok, TokenList):
            yield from _iter_all_tokens(tok)
