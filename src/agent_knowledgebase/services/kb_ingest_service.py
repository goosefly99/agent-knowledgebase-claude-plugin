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

import time
import uuid
from typing import TYPE_CHECKING, Any, Iterable, Optional

from agent_knowledgebase.models import Source, SourceType
from agent_knowledgebase.services.dedup_service import DEFAULT_DEDUP_POLICY, DedupPolicy
from agent_knowledgebase.services.stderr_log import knowledgebase_stderr_log

if TYPE_CHECKING:
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService

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


def normalize_dedup_policy(raw: Optional[str]) -> DedupPolicy:
    """Parse and normalise a raw dedup-policy string.

    Accepts ``"force-add"`` (hyphen) as an alias for ``"force_add"``
    (underscore) so callers don't need to remember which separator is
    canonical.  Returns :data:`DEFAULT_DEDUP_POLICY` when *raw* is
    ``None`` or empty.

    Raises:
        ValueError: If *raw* is a non-empty string that names no known policy.
    """
    if not raw:
        return DEFAULT_DEDUP_POLICY
    normalised = raw.replace("-", "_")
    try:
        return DedupPolicy(normalised)
    except ValueError:
        valid = [p.value for p in DedupPolicy]
        raise ValueError(
            f"Unknown dedup_policy {raw!r}. Valid values: {valid} (hyphens accepted for 'force-add')."
        )


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


# ---------------------------------------------------------------------------
# Batch entry point (v0.6.0)
# ---------------------------------------------------------------------------


def run_ingest_batch(
    service: "KnowledgebaseService",
    kb_id: str,
    source_defs: list[dict[str, Any]],
    default_policy: DedupPolicy = DEFAULT_DEDUP_POLICY,
    request_id: Optional[str] = None,
    tool_caller_version: Optional[str] = None,
) -> list[Source]:
    """Execute a batch ingest with structured stderr telemetry.

    Emits one ``ingest_batch_start`` stderr line before the loop, one
    ``ingest_source`` line per source outcome, and one
    ``ingest_batch_end`` line with summed counters after the loop.

    A ``request_id`` is synthesized when not supplied so every
    PipelineRun telemetry row carries one.

    Args:
        service: The ``KnowledgebaseService`` that owns the
            knowledgebase. Used only for its ``ingest_source`` method
            so this function is trivially mockable in tests.
        kb_id: Target knowledgebase id.
        source_defs: Already-decoded list of source specification dicts
            (each with at minimum ``source_type`` and ``uri`` keys and
            optionally ``metadata``, ``dedup_key`` and
            ``dedup_policy``).
        default_policy: Top-level default dedup policy used when a
            row-level ``dedup_policy`` is not provided.
        request_id: Caller-supplied correlation id. A new uuid4 hex is
            synthesized when this is ``None`` so the telemetry row is
            always populated.
        tool_caller_version: Caller-supplied semver string; may be
            ``None``.

    Returns:
        List of :class:`Source` results from each ``ingest_source``
        call, in input order.
    """
    if request_id is None:
        request_id = uuid.uuid4().hex

    default_policy_value = default_policy.value if isinstance(
        default_policy, DedupPolicy
    ) else str(default_policy)

    batch_size = len(source_defs)
    batch_start = time.monotonic()
    knowledgebase_stderr_log(
        kb_id=kb_id,
        op="ingest_batch_start",
        phase="pre_run",
        elapsed_ms=0,
        rows_in=batch_size,
        rows_ok=0,
        rows_skipped=0,
        rows_failed=0,
        dedup_policy=default_policy_value,
        request_id=request_id,
        tool_caller_version=tool_caller_version,
    )

    results: list[Source] = []
    rows_ok = 0
    rows_skipped = 0
    rows_failed = 0

    for item in source_defs:
        st = SourceType(item["source_type"])
        uri = item["uri"]
        meta = item.get("metadata", {})
        row_dedup_key = item.get("dedup_key") or None
        row_policy_raw = item.get("dedup_policy")
        row_policy = normalize_dedup_policy(row_policy_raw) if row_policy_raw else default_policy
        row_policy_value = row_policy.value if isinstance(
            row_policy, DedupPolicy
        ) else str(row_policy)

        item_start = time.monotonic()
        # Snapshot existing source id (if any) to detect skip vs. replace vs. insert.
        ctx = service._ctx(kb_id)  # noqa: SLF001 — internal plumbing by design
        pre_existing = (
            ctx.db.find_source_by_dedup_key(kb_id, row_dedup_key)
            if row_dedup_key
            else None
        )

        try:
            source = service.ingest_source(
                kb_id,
                st,
                uri,
                meta,
                dedup_key=row_dedup_key,
                dedup_policy=row_policy,
                request_id=request_id,
                tool_caller_version=tool_caller_version,
                batch_size=batch_size,
            )
            results.append(source)
            # Classify the outcome for the per-source telemetry line.
            if (
                pre_existing is not None
                and row_policy == DedupPolicy.skip
                and source.id == pre_existing.id
            ):
                op = "ingest_source"
                phase = "post_run"
                row_ok_delta = 0
                row_skipped_delta = 1
                row_failed_delta = 0
            else:
                op = "ingest_source"
                phase = "finalize"
                row_ok_delta = 1
                row_skipped_delta = 0
                row_failed_delta = 0

            rows_ok += row_ok_delta
            rows_skipped += row_skipped_delta
            rows_failed += row_failed_delta
            elapsed_ms = int((time.monotonic() - item_start) * 1000)
            knowledgebase_stderr_log(
                kb_id=kb_id,
                op=op,
                phase=phase,
                elapsed_ms=elapsed_ms,
                rows_in=1,
                rows_ok=row_ok_delta,
                rows_skipped=row_skipped_delta,
                rows_failed=row_failed_delta,
                dedup_policy=row_policy_value,
                request_id=request_id,
                tool_caller_version=tool_caller_version,
            )
        except Exception as exc:
            rows_failed += 1
            elapsed_ms = int((time.monotonic() - item_start) * 1000)
            knowledgebase_stderr_log(
                kb_id=kb_id,
                op="ingest_source",
                phase="post_run",
                elapsed_ms=elapsed_ms,
                rows_in=1,
                rows_ok=0,
                rows_skipped=0,
                rows_failed=1,
                dedup_policy=row_policy_value,
                request_id=request_id,
                tool_caller_version=tool_caller_version,
                error_code="INGEST_SOURCE_FAILED",
                error_message=str(exc),
            )
            raise

    total_elapsed_ms = int((time.monotonic() - batch_start) * 1000)
    knowledgebase_stderr_log(
        kb_id=kb_id,
        op="ingest_batch_end",
        phase="post_run",
        elapsed_ms=total_elapsed_ms,
        rows_in=batch_size,
        rows_ok=rows_ok,
        rows_skipped=rows_skipped,
        rows_failed=rows_failed,
        dedup_policy=default_policy_value,
        request_id=request_id,
        tool_caller_version=tool_caller_version,
    )
    return results
