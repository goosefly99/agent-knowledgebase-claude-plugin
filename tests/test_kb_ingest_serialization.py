"""Tests for per-kb_id serialization around ingest_source.

Verifies three properties:
  1. Two concurrent ingest calls targeting the *same* kb_id execute serially
     (no overlap in their pipeline execution intervals).
  2. Two concurrent ingest calls targeting *different* kb_ids execute in
     parallel (their intervals overlap).
  3. The [kb-lock] acquired / released log lines appear on stderr for each
     ingest call, tagged with the correct kb_id.

Test strategy
-------------
Tests 1 and 2 need to observe concurrency (or its absence) at the lock level.
To avoid fighting SQLite's same-thread restriction (the connection is opened
in the main thread; worker threads can't reuse it), we mock
``_ingest_source_locked`` — the inner pipeline body — with a function that
records its execution interval via a ``threading.Event`` gate.  This lets the
test exercise the real locking logic in ``ingest_source`` without touching
SQLite at all.

Test 3 uses a real KnowledgebaseService backed by a ``tmp_path`` SQLite
database (same pattern as ``test_kb_ingest_service.py::test_dedup_policy_skip``)
to verify end-to-end stderr output.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.models import Chunk, SourceType
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# ---------------------------------------------------------------------------
# Helper dataclass for tracking call intervals
# ---------------------------------------------------------------------------


@dataclass
class _Interval:
    start: float
    end: float

    def overlaps(self, other: "_Interval") -> bool:
        """Return True if the two intervals have any overlap."""
        return self.start < other.end and other.start < self.end


# ---------------------------------------------------------------------------
# Test 1 — serial execution on the same kb_id
# ---------------------------------------------------------------------------


def test_same_kb_id_executes_serially() -> None:
    """Two concurrent ingest calls on the same kb_id must not overlap.

    We mock ``_ingest_source_locked`` so the test only exercises the locking
    logic in ``ingest_source``, not the full pipeline (which requires a
    same-thread SQLite connection).
    """
    svc = KnowledgebaseService.__new__(KnowledgebaseService)
    # Initialise only the lock attributes used by ingest_source.
    import threading as _threading
    svc._kb_locks: dict = {}
    svc._kb_locks_guard = _threading.Lock()

    intervals: list[_Interval] = []
    errors: list[Exception] = []

    # Gate: both threads wait here before entering, maximising lock contention.
    ready = threading.Event()

    def _slow_locked(kb_id, *args, **kwargs):
        """Mock pipeline body: record the execution interval and sleep."""
        # Record start INSIDE the lock body so the interval reflects only
        # the time the lock was actually held — not the wait time.
        t0 = time.monotonic()
        time.sleep(0.15)
        t1 = time.monotonic()
        intervals.append(_Interval(t0, t1))
        return MagicMock()  # pretend it returned a Source

    with patch.object(svc, "_ingest_source_locked", side_effect=_slow_locked):

        def _call() -> None:
            ready.wait()
            try:
                svc.ingest_source("kb-A", SourceType.file, "/tmp/x.txt")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_call)
        t2 = threading.Thread(target=_call)
        t1.start()
        t2.start()
        ready.set()
        t1.join(timeout=10)
        t2.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert len(intervals) == 2

    i0, i1 = intervals
    assert not i0.overlaps(i1), (
        f"Intervals overlapped — serialization lock is missing or broken. "
        f"interval[0]={i0}, interval[1]={i1}"
    )


# ---------------------------------------------------------------------------
# Test 2 — parallel execution on different kb_ids
# ---------------------------------------------------------------------------


def test_different_kb_ids_execute_in_parallel() -> None:
    """Two concurrent ingest calls on different kb_ids must overlap.

    Same mocking strategy as test 1.
    """
    svc = KnowledgebaseService.__new__(KnowledgebaseService)
    import threading as _threading
    svc._kb_locks: dict = {}
    svc._kb_locks_guard = _threading.Lock()

    intervals: list[_Interval] = []
    errors: list[Exception] = []

    ready = threading.Event()

    def _slow_locked(kb_id, *args, **kwargs):
        # Record start inside the lock body so interval reflects actual execution.
        t0 = time.monotonic()
        time.sleep(0.15)
        t1 = time.monotonic()
        intervals.append(_Interval(start=t0, end=t1))
        return MagicMock()

    with patch.object(svc, "_ingest_source_locked", side_effect=_slow_locked):

        def _call(kb_id: str) -> None:
            ready.wait()
            try:
                svc.ingest_source(kb_id, SourceType.file, "/tmp/x.txt")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_call, args=("kb-A",))
        t2 = threading.Thread(target=_call, args=("kb-B",))
        t1.start()
        t2.start()
        ready.set()
        t1.join(timeout=10)
        t2.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert len(intervals) == 2

    i0, i1 = intervals
    assert i0.overlaps(i1), (
        f"Intervals did not overlap — different-kb_id calls are being serialized "
        f"(global lock bug). interval[0]={i0}, interval[1]={i1}"
    )


# ---------------------------------------------------------------------------
# Test 3 — stderr log lines (full-stack, real SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _log_svc(test_config: Settings) -> KnowledgebaseService:
    """KnowledgebaseService with mocked embedder, vectorstore, and ingestion."""
    service = KnowledgebaseService(test_config)
    service._embedder_instance = MagicMock()
    service._embedder_instance.embed.side_effect = lambda texts: [[0.1, 0.2]] * len(texts)
    service._embedder_instance.model_name = "test-embedder"
    service._get_vectorstore = lambda kb_id: MagicMock(  # type: ignore[assignment]
        add=MagicMock(return_value=None),
        delete=MagicMock(return_value=None),
    )
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.side_effect = lambda *_, **__: [
        Chunk(source_id="", kb_id="", content="hello world", metadata={})
    ]
    service._ingestion = mock_ingestion
    return service


def test_ingest_logs_lock_acquire_and_release(
    _log_svc: KnowledgebaseService, capsys: pytest.CaptureFixture
) -> None:
    """ingest_source must emit structured lock-lifecycle JSON lines to stderr.

    v0.6.0 routes the legacy ``[kb-lock] ...`` diagnostics through
    ``knowledgebase_stderr_log`` — each line is single-line JSON with at
    minimum ``kb_id``, ``op`` (``lock_acquired`` / ``lock_release``), and
    ``phase="lock"``.
    """
    import json as _json

    kb = _log_svc.create_kb("log-kb")
    _log_svc.ingest_source(kb.id, SourceType.file, "/tmp/log-test.txt")

    captured = capsys.readouterr()
    stderr = captured.err

    # Parse every JSON line on stderr and look for the acquired + release rows
    # tagged with our kb id.
    parsed: list[dict] = []
    for raw_line in stderr.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed.append(_json.loads(line))
        except _json.JSONDecodeError:
            # Not a structured log line (e.g. warning from other subsystems) — skip.
            continue
    ops = {(rec.get("kb_id"), rec.get("op")) for rec in parsed}
    assert (kb.id, "lock_acquired") in ops, (
        f"Expected lock_acquired JSON line for kb_id={kb.id}. Got stderr:\n{stderr}"
    )
    assert (kb.id, "lock_release") in ops, (
        f"Expected lock_release JSON line for kb_id={kb.id}. Got stderr:\n{stderr}"
    )


# ---------------------------------------------------------------------------
# Test 4 — update_source serializes with concurrent ingest_source
# ---------------------------------------------------------------------------


def test_update_source_serializes_with_ingest() -> None:
    """update_source and a concurrent ingest_source on the same kb_id must not overlap.

    update_source must hold the kb_id lock for cleanup + re-ingest so that
    a racing ingest_source cannot observe the torn state between chunk
    deletion and chunk insertion.

    Mocking strategy: mirror the bare-service approach used by tests 1 and 2.
    ``_ingest_source_locked`` is mocked to record execution intervals.
    ``db.delete_chunks_by_source`` and ``vectorstore.delete`` are mocked so
    the cleanup path in update_source doesn't touch SQLite.
    """
    import threading as _threading

    svc = KnowledgebaseService.__new__(KnowledgebaseService)
    svc._kb_locks: dict = {}
    svc._kb_locks_guard = _threading.Lock()

    kb_id = "kb-update-test"
    source_id = "src-001"

    # Build a minimal fake source with the fields update_source reads.
    from agent_knowledgebase.models import Source, SourceStatus

    fake_source = Source(
        kb_id=kb_id,
        source_type=SourceType.file,
        uri="/tmp/update-test.txt",
        metadata={},
        status=SourceStatus.ingested,
    )
    fake_source.id = source_id  # fix the id so we can look it up

    # Minimal fake DB: get_source, list_chunks, delete_chunks_by_source
    mock_db = MagicMock()
    mock_db.get_source.return_value = fake_source
    mock_db.list_chunks.return_value = []  # no old chunks to delete from vectorstore

    # Minimal fake context: wrap the mock db in a _KBContext-shaped object.
    mock_ctx = MagicMock()
    mock_ctx.db = mock_db

    # _find_context_by_source must return our fake context and the kb_id.
    svc._find_context_by_source = MagicMock(return_value=(mock_ctx, kb_id))  # type: ignore[assignment]

    # _get_vectorstore — not called because list_chunks returns [] but mock it anyway.
    svc._get_vectorstore = MagicMock(return_value=MagicMock())  # type: ignore[assignment]

    intervals: list[_Interval] = []
    errors: list[Exception] = []

    ready = threading.Event()
    # Gate: thread 2 (ingest) should start just AFTER thread 1 (update) grabs the lock.
    # We use a separate event to let thread 1 signal that it has acquired the lock.
    lock_acquired = threading.Event()

    def _slow_locked(kb_id_arg: str, *args: object, **kwargs: object) -> MagicMock:
        """Mock pipeline body: record interval and sleep."""
        t0 = time.monotonic()
        time.sleep(0.15)
        t1 = time.monotonic()
        intervals.append(_Interval(t0, t1))
        # Signal after the first call enters the body (update_source's re-ingest).
        if not lock_acquired.is_set():
            lock_acquired.set()
        return MagicMock()

    with patch.object(svc, "_ingest_source_locked", side_effect=_slow_locked):

        def _run_update() -> None:
            ready.wait()
            try:
                svc.update_source(source_id)
            except Exception as exc:
                errors.append(exc)

        def _run_ingest() -> None:
            # Wait until update_source has grabbed the lock before racing in.
            lock_acquired.wait(timeout=5)
            try:
                svc.ingest_source(kb_id, SourceType.file, "/tmp/race.txt")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_run_update)
        t2 = threading.Thread(target=_run_ingest)
        t1.start()
        t2.start()
        ready.set()
        t1.join(timeout=10)
        t2.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert len(intervals) == 2, f"Expected 2 pipeline executions, got {len(intervals)}"

    i0, i1 = intervals
    assert not i0.overlaps(i1), (
        f"Intervals overlapped — update_source lock coverage is missing or broken. "
        f"interval[0]={i0}, interval[1]={i1}"
    )
