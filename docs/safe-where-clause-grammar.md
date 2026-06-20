# Safe `row_selector` grammar for `kb_ingest_batch`

This document is the canonical allow-list for WHERE clauses passed as
`row_selector` to `kb_ingest_batch` with `source_type="sql_database"`.
It is the contract agent-knowledgebase enforces on untrusted SQL text.
The allow-list is enforced by the sqlparse-AST validator
(`agent-knowledgebase/src/agent_knowledgebase/services/where_clause_validator.py`)
and layered behind the SQLite `mode=ro` URI rewrite (see line 70 of
`src/agent_knowledgebase/ingestors/sql_database.py`) for
defense-in-depth. Downstream repos that need a user-facing reference
(for example the orchestrator's ETL skills) should link here rather
than redefine the grammar.

## Allow-list grammar

```ebnf
row_selector    ::= expr
expr            ::= or_expr
or_expr         ::= and_expr  ( "OR"  and_expr )*
and_expr        ::= not_expr  ( "AND" not_expr )*
not_expr        ::= "NOT" not_expr
                  | predicate
predicate       ::= comparison
                  | null_check
                  | membership
                  | like_match
                  | "(" expr ")"
comparison      ::= identifier cmp_op  literal
                  | identifier cmp_op  identifier
cmp_op          ::= "=" | "!=" | "<>" | "<" | ">" | "<=" | ">="
null_check      ::= identifier "IS" "NULL"
                  | identifier "IS" "NOT" "NULL"
membership      ::= identifier [ "NOT" ] "IN" "(" literal ( "," literal )* ")"
like_match      ::= identifier [ "NOT" ] "LIKE" string_literal
                  [ "ESCAPE" string_literal ]

identifier      ::= /[A-Za-z_][A-Za-z0-9_]*/          ; bare only
literal         ::= string_literal | number_literal | "NULL"
string_literal  ::= "'" any-char-with-doubled-quotes "'"
number_literal  ::= integer | float
```

### Notes

- Identifiers are **not table-qualified**. `table.col` is rejected in
  the current release; table-qualified identifiers are reserved for
  v0.4.0 scoping work.
- String literals are single-quoted. A literal apostrophe must be
  doubled (`'O''Brien'`). Double-quoted identifiers are tolerated by
  the parser but not required.
- `NULL` is accepted as a literal only inside `IS NULL` / `IS NOT NULL`
  — write null-tests with the null-check form, not `col = NULL`.
- Parenthesization `( expr )` is accepted anywhere an expression is
  allowed. Parens do not introduce subqueries — a `SELECT` token
  inside any parenthesized group triggers
  `WHERE_SUBQUERY_FORBIDDEN`.

## Deny-list

Every construct below is rejected with a
`WhereClauseValidationError` (stable error code in parentheses):

- Subqueries — `SELECT ... FROM ...` inside parens (`WHERE_SUBQUERY_FORBIDDEN`).
- Function calls — `LOWER(col)`, `COALESCE(a, b)`, `COUNT(*)`,
  `JSON_EXTRACT(...)`, any `name(...)` shape
  (`WHERE_FUNCTION_CALL_FORBIDDEN`).
- Set operators — `UNION`, `INTERSECT`, `EXCEPT`
  (`WHERE_SET_OPERATOR_FORBIDDEN`).
- CTEs, window functions, `OVER (...)` clauses
  (`WHERE_DISALLOWED_KEYWORD`).
- Column aliases — `col AS name` inside the `row_selector`
  (`WHERE_DISALLOWED_KEYWORD`).
- Table-qualified names — `t.col`, `schema.table.col` (reserved for
  v0.4.0; currently rejected at parse time).
- Multi-statement — any semicolon `;` in the non-literal portion of
  the input (`WHERE_MULTI_STATEMENT`).
- `LIMIT`, `OFFSET`, `ORDER BY`, `GROUP BY`, `HAVING` — these belong
  on the outer query, not the `row_selector`
  (`WHERE_DISALLOWED_KEYWORD`).
- SQL comments — `-- ...` line comments and `/* ... */` block
  comments (`WHERE_CONTAINS_COMMENT`).
- Backtick-quoted identifiers `` `col` `` — use double quotes if you
  must quote an identifier (`WHERE_BACKTICK_FORBIDDEN`).
- SQLite control-plane keywords — `PRAGMA`, `ATTACH`, `DETACH`
  (`WHERE_PRAGMA_FORBIDDEN`, `WHERE_ATTACH_FORBIDDEN`,
  `WHERE_DETACH_FORBIDDEN`).
- DML / DDL keywords in any position — `INSERT`, `UPDATE`, `DELETE`,
  `DROP`, `CREATE`, `ALTER`, `TRUNCATE`, `GRANT`, `REVOKE`, `EXEC`,
  `VACUUM`, `REINDEX`, etc. (`WHERE_DISALLOWED_KEYWORD`).
- Arithmetic and bitwise operators — `+`, `-`, `*`, `/`, `%`, `|`,
  `&`, `^`, `||` (string concat) (`WHERE_ARITHMETIC_FORBIDDEN`).
- Bind parameters — `?`, `:name`, `$1` (`WHERE_DISALLOWED_KEYWORD`).
- Wildcards `*` outside a string literal (`WHERE_DISALLOWED_OPERATOR`).
- `CASE` / `WHEN` / `THEN` / `ELSE` / `END` expressions
  (`WHERE_DISALLOWED_KEYWORD`).
- `BETWEEN ... AND ...` — rewrite as
  `col >= a AND col <= b` (`WHERE_DISALLOWED_KEYWORD`).
- `CAST(expr AS type)` (`WHERE_DISALLOWED_KEYWORD`).

## Example filters

All ten examples below have been verified against the current
validator. Copy them verbatim into your `row_selector`.

### 1. Select a single row by primary key

```sql
id = 42
```

Use this when you know the exact row you want to ingest.

### 2. Select a small set of IDs (chained `OR`)

```sql
id = 1 OR id = 2 OR id = 3
```

Idiomatic for two or three IDs. For larger lists prefer `IN (...)`
as shown in example 10's style — it is also accepted by the
validator.

### 3. Date range

```sql
created_at >= '2026-01-01' AND created_at < '2026-04-01'
```

Half-open date range — common shape for time-windowed ingest.

### 4. Filter by author

```sql
author = 'alice'
```

One-column equality filter on a text column.

### 5. Date range combined with author

```sql
created_at >= '2026-04-01' AND author = 'bob'
```

Combines a date predicate with an equality filter.

### 6. Exclude archived rows

```sql
status != 'archived'
```

Inequality against a text column; also accepted as `status <> 'archived'`.

### 7. Filter by tag

```sql
tag = 'research'
```

Trivial equality filter; useful when the source table has a single
`tag` column per row.

### 8. Numeric comparison

```sql
word_count > 500
```

Numeric comparisons work for `INTEGER` and `REAL` / `FLOAT` columns
equally.

### 9. Null check on soft-delete column

```sql
deleted_at IS NULL
```

Null-check form — the only accepted way to test for `NULL`. Do not
write `deleted_at = NULL`; SQL compares `NULL` to `NULL` as `NULL`,
not true.

### 10. Compound `AND` / `OR` with parens

```sql
(status = 'published' OR status = 'featured') AND author != 'bot'
```

Parens are accepted for grouping. The content inside parens must
itself be a valid expression — it cannot contain a `SELECT`, a
function call, or any other denied construct.

## Defense-in-depth

Even a correct allow-list validator can have bugs or miss an edge
case. Two layers protect against the consequences.

**Layer 1 — sqlparse AST validation.** The validator first runs a
literal-aware string-level pass (rejecting comments, semicolons,
backticks, `PRAGMA` / `ATTACH` / `DETACH` / `UNION` / `INTERSECT` /
`EXCEPT`) and then walks the parsed AST, rejecting any token category
not on the allow-list above. Every rejection raises a
`WhereClauseValidationError` with a stable `code` and an
`offending_token` for remediation.

**Layer 2 — SQLite `mode=ro` URI rewrite.** Even if the validator
misses an edge case, the SQLAlchemy engine for every sqlite source is
constructed via a URI rewrite that pins the connection physically
read-only: `sqlite:///file:<path>?mode=ro&uri=true`. The rewrite
lives at line 70 of
`src/agent_knowledgebase/ingestors/sql_database.py`
(`_build_readonly_sqlite_url`). Any attempted `INSERT` / `UPDATE` /
`DELETE` / `DROP` against the engine — whether through a smuggled
token or a future parser bug — surfaces as
`sqlite3.OperationalError: attempt to write a readonly database`
before touching the file.

## Validator reference

- **File:** `src/agent_knowledgebase/services/where_clause_validator.py`
- **Entry point:** `validate_where_clause(where: str) -> None`
- **Error type:** `WhereClauseValidationError`
- **Stable error codes:** the `CODE_*` module-level constants (e.g.
  `CODE_EMPTY`, `CODE_COMMENT`, `CODE_SUBQUERY`,
  `CODE_FUNCTION_CALL`, `CODE_ARITHMETIC`, `CODE_DISALLOWED_KEYWORD`,
  `CODE_DISALLOWED_OPERATOR`, `CODE_PARSE_FAILED`).

When you add a new accepted shape, treat this document as the source
of truth: update the grammar block, add a worked example, and confirm
the validator still accepts it. The validator's pre-parse guards
(`_reject_backticks`, `_reject_raw_comments`, `_reject_multi_statement`,
`_reject_forbidden_keywords`) run before the AST walk; keep both in
sync when the grammar changes.
