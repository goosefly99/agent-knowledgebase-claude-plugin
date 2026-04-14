"""Ingest-service guardrail tests (batch-size cap, sequential-exec, dedup).

The first guardrail covered here is the release-1 hard-reject of
oversized ``source_type='sql_database'`` batches.  ROADMAP.md names
this module as the acceptance surface:

    tests/test_kb_ingest_service.py
      -- (a) 51-row rejection,
      -- (b) per-kb_id serialization via overlapping async calls,
      -- (c) each of skip/replace/force-add dedup policies.

Only (a) lands in this commit -- the sequential-exec and dedup
items are separate blocking roadmap lines.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_knowledgebase.server import kb_ingest_batch
from agent_knowledgebase.services.kb_ingest_service import (
    CODE_BATCH_SIZE_EXCEEDED,
    MAX_SQL_DATABASE_BATCH_SIZE,
    BatchSizeExceededError,
    validate_batch_size,
)


# ---------------------------------------------------------------------------
# Unit-level: the validator itself
# ---------------------------------------------------------------------------


def test_max_sql_database_batch_size_is_fifty() -> None:
    """The cap is fixed by spec; regression-gate the value."""
    assert MAX_SQL_DATABASE_BATCH_SIZE == 50


def test_validate_batch_size_accepts_exactly_fifty() -> None:
    """The boundary value (50 rows) must not raise."""
    rows = [{"source_type": "sql_database", "uri": f"sqlite:///x{i}"} for i in range(50)]
    # No exception == accept.
    assert validate_batch_size(rows) is None


def test_validate_batch_size_rejects_fifty_one() -> None:
    """51 rows is the first rejected count."""
    rows = [{"source_type": "sql_database", "uri": f"sqlite:///x{i}"} for i in range(51)]
    with pytest.raises(BatchSizeExceededError) as exc:
        validate_batch_size(rows)
    assert exc.value.code == CODE_BATCH_SIZE_EXCEEDED
    assert exc.value.received == 51
    assert exc.value.limit == 50
    assert exc.value.source_type == "sql_database"
    # Remediation text must name the limit and tell the caller to split.
    assert "50" in exc.value.message
    assert "split" in exc.value.message.lower()


def test_validate_batch_size_ignores_non_sql_rows() -> None:
    """Non-sql source_types do not count toward the sql_database cap."""
    rows = [{"source_type": "file", "uri": f"/tmp/f{i}"} for i in range(200)]
    # No exception -- file ingests are not capped.
    assert validate_batch_size(rows) is None


def test_validate_batch_size_mixed_batch_under_cap() -> None:
    """Mixed batches: only the sql_database slice counts."""
    rows = (
        [{"source_type": "sql_database", "uri": "sqlite:///a"} for _ in range(50)]
        + [{"source_type": "file", "uri": "/tmp/x"} for _ in range(100)]
    )
    assert validate_batch_size(rows) is None


def test_validate_batch_size_mixed_batch_over_cap() -> None:
    """Mixed batch with 51 sql rows is rejected even if total is lower."""
    rows = (
        [{"source_type": "sql_database", "uri": "sqlite:///a"} for _ in range(51)]
        + [{"source_type": "file", "uri": "/tmp/x"}]
    )
    with pytest.raises(BatchSizeExceededError) as exc:
        validate_batch_size(rows)
    assert exc.value.received == 51


def test_validate_batch_size_tolerates_malformed_entries() -> None:
    """Non-dict entries don't count -- the server layer surfaces its own error."""
    rows = ["not-a-dict", 42, None]
    # No sql_database rows -> accept.  The non-dict entries trigger a
    # downstream error (SourceType conversion), not this guard.
    assert validate_batch_size(rows) is None


def test_batch_size_error_is_valueerror_subclass() -> None:
    """Existing MCP error-surfacing catches ``ValueError``; the new error must match."""
    err = BatchSizeExceededError(received=51)
    assert isinstance(err, ValueError)
    assert err.code == CODE_BATCH_SIZE_EXCEEDED


# ---------------------------------------------------------------------------
# Acceptance: the ROADMAP.md 51-row rejection test.
#
# From ROADMAP.md:
#     ``kb_ingest_batch`` with 51 rows of ``source_type=sql_database``
#     returns a structured error and zero rows written.
#
# We assert the structured error via ``BatchSizeExceededError`` attributes
# and assert zero rows written by confirming the service's
# ``ingest_source`` was never called -- the guard runs before any
# service dispatch, so nothing can reach the DB.
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_service():
    return MagicMock()


@pytest.fixture(autouse=True)
def _patch_get_service(mock_service):
    with patch("agent_knowledgebase.server._get_service", return_value=mock_service):
        yield


def test_kb_ingest_batch_rejects_51_sql_database_rows_with_zero_writes(
    mock_service: MagicMock,
) -> None:
    """ROADMAP acceptance: 51 sql_database rows -> structured error, zero writes."""
    sources = json.dumps(
        [
            {"source_type": "sql_database", "uri": f"sqlite:///x{i}"}
            for i in range(51)
        ]
    )

    with pytest.raises(BatchSizeExceededError) as exc:
        kb_ingest_batch("kb-1", sources)

    # Structured-error contract: stable code + remediation-bearing message.
    assert exc.value.code == CODE_BATCH_SIZE_EXCEEDED
    assert exc.value.received == 51
    assert exc.value.limit == 50
    assert "split" in exc.value.message.lower()

    # Zero rows written: the service was never dispatched to.
    mock_service.ingest_source.assert_not_called()


def test_kb_ingest_batch_accepts_exactly_50_sql_database_rows(
    mock_service: MagicMock,
) -> None:
    """The boundary (50 rows) proceeds to the service layer."""
    fake_source = MagicMock()
    fake_source.model_dump_json.return_value = "{}"
    mock_service.ingest_source.return_value = fake_source

    sources = json.dumps(
        [
            {"source_type": "sql_database", "uri": f"sqlite:///x{i}"}
            for i in range(50)
        ]
    )
    # Should not raise -- the guard permits exactly the cap.
    result = kb_ingest_batch("kb-1", sources)
    assert mock_service.ingest_source.call_count == 50
    # Result is a valid JSON array string.
    assert result.startswith("[") and result.endswith("]")
