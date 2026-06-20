"""Structured stderr logging helper for agent-knowledgebase operations.

Emits single-line JSON records to ``sys.stderr`` describing lifecycle
events within the ingestion pipeline (batch boundaries, per-source
outcomes, lock acquisition/release, and failure telemetry).

The helper is intentionally stdlib-only — no new runtime dependency is
introduced. Output is a single JSON document terminated by a single
newline (the one produced by ``print(..., file=sys.stderr)``) and the
stream is flushed immediately so parent processes (e.g. an MCP client
or test harness using ``capsys``) see the line as soon as the call
returns.

Field contract (all required keys appear on every emission):

* ``kb_id``              — target knowledgebase id
* ``op``                 — short operation label (e.g.
  ``"ingest_batch_start"``, ``"ingest_source"``, ``"lock_acquire"``,
  ``"lock_acquired"``, ``"lock_release"``)
* ``phase``              — a ``PipelinePhase`` value when the emission
  originates inside a named pipeline phase, or ``"pre_run"`` /
  ``"post_run"`` / ``"lock"`` for non-phase lifecycle events
* ``elapsed_ms``         — integer milliseconds; zero is legal for
  instantaneous events (e.g. lock acquire announce)
* ``rows_in``            — input count (number of sources in the batch,
  or ``1`` for a single-source emission, or ``0`` for non-row events)
* ``rows_ok``            — sources that ingested successfully
* ``rows_skipped``       — sources that were skipped (dedup)
* ``rows_failed``        — sources that errored
* ``dedup_policy``       — a ``DedupPolicy`` value or ``"n/a"`` when the
  emission doesn't carry dedup semantics
* ``request_id``         — caller-supplied correlation id or ``None``
* ``tool_caller_version``— caller-supplied semver string or ``None``

``error_code`` (SCREAMING_SNAKE_CASE) and ``error_message`` are
included only when the caller passes them (i.e. on failure emissions).
"""

from __future__ import annotations

import json
import sys


def knowledgebase_stderr_log(
    *,
    kb_id: str,
    op: str,
    phase: str,
    elapsed_ms: int,
    rows_in: int,
    rows_ok: int,
    rows_skipped: int,
    rows_failed: int,
    dedup_policy: str,
    request_id: str | None,
    tool_caller_version: str | None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    """Emit a single-line JSON event to ``sys.stderr``.

    See module docstring for the full field contract. ``error_code`` and
    ``error_message`` are included in the output only when non-``None``
    — success emissions stay lean.

    The call is synchronous and flushes stderr immediately.
    """
    payload: dict[str, object] = {
        "kb_id": kb_id,
        "op": op,
        "phase": phase,
        "elapsed_ms": elapsed_ms,
        "rows_in": rows_in,
        "rows_ok": rows_ok,
        "rows_skipped": rows_skipped,
        "rows_failed": rows_failed,
        "dedup_policy": dedup_policy,
        "request_id": request_id,
        "tool_caller_version": tool_caller_version,
    }
    if error_code is not None:
        payload["error_code"] = error_code
    if error_message is not None:
        payload["error_message"] = error_message
    print(json.dumps(payload), file=sys.stderr, flush=True)
