"""Ingest-service guardrails for ``kb_ingest`` / ``kb_ingest_batch``.

This module owns pre-dispatch policy for batch ingest. The first
guardrail landing here is the release-1 hard-reject of oversized
``source_type='sql_database'`` batches: any batch with more than
:data:`MAX_SQL_DATABASE_BATCH_SIZE` rows raises
:class:`BatchSizeExceededError` *before* any source record is created
or any ingestor runs, so the on-disk DB stays untouched.

The 50-row cap is fixed by spec (validated against chromadb's
soft-performance threshold; see ROADMAP.md "Accepted as-is").  The
release-1 posture is hard-reject, not soft-warn -- the round-3
debate synthesizer overturned the spec's soft-warn proposal on the
grounds that a month of production traffic would build dependencies
on oversized batches before release 2 flipped the switch.

Callers (``server.py:kb_ingest_batch``) invoke
:func:`validate_batch_size` with the decoded ``sources`` list before
iterating; the function returns ``None`` on accept and raises on
reject.  The error surfaces at the MCP tool boundary with a stable
``code`` and human-readable ``message`` that tells the caller to
split into batches of at most :data:`MAX_SQL_DATABASE_BATCH_SIZE`.
"""

from __future__ import annotations

from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Maximum number of ``source_type='sql_database'`` rows accepted in a
#: single ``kb_ingest_batch`` call.  Fixed per spec -- do not tune without
#: re-running validation against chromadb's soft-performance threshold.
MAX_SQL_DATABASE_BATCH_SIZE: int = 50

#: Stable error code embedded in the structured error raised on over-batch.
CODE_BATCH_SIZE_EXCEEDED: str = "BATCH_SIZE_EXCEEDED"


# ---------------------------------------------------------------------------
# Structured error
# ---------------------------------------------------------------------------


class BatchSizeExceededError(ValueError):
    """Raised when a ``kb_ingest_batch`` call exceeds the SQL-database row cap.

    Subclasses :class:`ValueError` so existing MCP error-surfacing paths
    serialize it the same way they serialize the other structured
    errors in this codebase (e.g.
    :class:`~agent_knowledgebase.services.where_clause_validator.WhereClauseValidationError`).

    Attributes:
        code: Stable machine-readable error code
            (:data:`CODE_BATCH_SIZE_EXCEEDED`).
        message: Human-readable remediation hint naming the limit and
            telling the caller to split the batch.
        limit: The cap enforced for this rejection
            (:data:`MAX_SQL_DATABASE_BATCH_SIZE`).
        received: The number of ``source_type='sql_database'`` rows the
            caller actually submitted.
        source_type: The source-type whose cap was exceeded
            (``'sql_database'`` in release 1; the field exists so future
            per-source-type caps can reuse this error envelope).
    """

    def __init__(
        self,
        received: int,
        limit: int = MAX_SQL_DATABASE_BATCH_SIZE,
        source_type: str = "sql_database",
    ) -> None:
        self.code = CODE_BATCH_SIZE_EXCEEDED
        self.limit = limit
        self.received = received
        self.source_type = source_type
        self.message = (
            f"kb_ingest_batch received {received} rows with "
            f"source_type={source_type!r}; the per-call cap is {limit}. "
            f"Split the input into batches of at most {limit} rows and "
            f"re-submit each batch."
        )
        super().__init__(f"[{self.code}] {self.message}")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _count_sql_database_rows(sources: Iterable[Any]) -> int:
    """Return the number of items in *sources* whose ``source_type`` is ``'sql_database'``.

    Non-dict entries and entries without a ``source_type`` key contribute
    zero to the count -- the caller's downstream validation will reject
    them with a more specific error (e.g.
    :class:`SourceType` conversion in the server layer).  This helper is
    deliberately permissive so the batch-size guard never shadows those
    more informative errors.
    """
    count = 0
    for item in sources:
        if not isinstance(item, dict):
            continue
        if item.get("source_type") == "sql_database":
            count += 1
    return count


def validate_batch_size(sources: Iterable[Any]) -> None:
    """Reject ``kb_ingest_batch`` inputs that exceed the SQL-database row cap.

    *sources* is the decoded payload of ``kb_ingest_batch``'s ``sources``
    JSON argument -- a list of source-definition dicts, each with at
    least a ``source_type`` key.

    The guard counts only rows with ``source_type='sql_database'`` and
    raises :class:`BatchSizeExceededError` if that count exceeds
    :data:`MAX_SQL_DATABASE_BATCH_SIZE`.  Mixed-type batches are
    permitted; only the sql_database slice is capped (non-sql rows can
    legitimately number in the thousands for e.g. per-file ingests).

    Args:
        sources: Decoded ``sources`` payload (list-like of dicts).

    Raises:
        BatchSizeExceededError: If more than
            :data:`MAX_SQL_DATABASE_BATCH_SIZE` rows have
            ``source_type='sql_database'``.  The error carries the
            observed count and the limit.
    """
    count = _count_sql_database_rows(sources)
    if count > MAX_SQL_DATABASE_BATCH_SIZE:
        raise BatchSizeExceededError(received=count)
