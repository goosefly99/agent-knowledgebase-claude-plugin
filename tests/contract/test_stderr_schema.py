"""Phase 2 contract test: knowledgebase_stderr_log 11-field schema.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        constraints[2] (knowledgebase_stderr_log JSON schema frozen
        across all backends).

Every emission of
:func:`agent_knowledgebase.services.stderr_log.knowledgebase_stderr_log`
must carry EXACTLY the 11 required fields — no extras (beyond the
opt-in ``error_code`` / ``error_message`` failure pair), no omissions.

The schema is pinned by capturing every emission during a representative
``kb_ingest_batch`` run and asserting each emitted JSON line matches
the contract. Capture is via monkeypatching the helper to redirect
emissions to a list — explicitly NOT ``capsys`` because the
single-worker tool executor in ``server.py`` runs tools on a thread
that is not the pytest main thread, and the nested timeout wrapper
muddles ``capsys`` output.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.models import Chunk
from agent_knowledgebase.services import stderr_log as stderr_log_module
from agent_knowledgebase.services.kb_ingest_service import run_ingest_batch
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# Required keys per spec.architecture.integration_points[2] and
# v0.6.0 CHANGELOG.
_REQUIRED_FIELDS = frozenset(
    {
        "kb_id",
        "op",
        "phase",
        "elapsed_ms",
        "rows_in",
        "rows_ok",
        "rows_skipped",
        "rows_failed",
        "dedup_policy",
        "request_id",
        "tool_caller_version",
    }
)

# Fields permitted only on failure emissions.
_OPTIONAL_ERROR_FIELDS = frozenset({"error_code", "error_message"})


@pytest.fixture()
def captured_emissions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Redirect every ``knowledgebase_stderr_log`` call into a list.

    Returns a mutable list that the test asserts against after the
    operation under test runs.

    The monkeypatch is scoped to ``services.stderr_log`` AND the
    re-exports in ``services.knowledgebase`` and
    ``services.kb_ingest_service`` since those modules imported the
    function by name at module-load time.
    """
    captured: list[dict[str, object]] = []

    def _capture(**kwargs: object) -> None:
        captured.append(dict(kwargs))

    monkeypatch.setattr(stderr_log_module, "knowledgebase_stderr_log", _capture)
    # The two callers imported the symbol directly:
    monkeypatch.setattr(
        "agent_knowledgebase.services.knowledgebase.knowledgebase_stderr_log",
        _capture,
    )
    monkeypatch.setattr(
        "agent_knowledgebase.services.kb_ingest_service.knowledgebase_stderr_log",
        _capture,
    )
    return captured


@pytest.fixture()
def chromadb_service(test_config: Settings) -> KnowledgebaseService:
    """Chromadb-backed service with mocked embedder/vectorstore/ingestion."""
    svc = KnowledgebaseService(test_config)
    svc._embedder_instance = MagicMock()
    svc._embedder_instance.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    svc._embedder_instance.model_name = "stderr-test-embedder"
    svc._get_vectorstore = lambda kb_id: MagicMock(  # type: ignore[assignment]
        add=MagicMock(return_value=None),
        delete=MagicMock(return_value=None),
        count=MagicMock(return_value=0),
    )
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.side_effect = lambda *_, **__: [
        Chunk(source_id="", kb_id="", content="stderr fixture", metadata={})
    ]
    svc._ingestion = mock_ingestion
    return svc


def _assert_schema_compliance(emissions: list[dict[str, object]]) -> None:
    """Assert every captured emission has exactly the 11 required keys.

    ``error_code`` / ``error_message`` are permitted only on failure
    emissions (we don't trigger a failure in the basic batch run, so
    those should never appear here).
    """
    assert emissions, "no emissions captured — fixture wiring is wrong"
    for idx, entry in enumerate(emissions):
        keys = set(entry)
        missing = _REQUIRED_FIELDS - keys
        assert not missing, (
            f"emission #{idx} missing required fields {sorted(missing)}: "
            f"{entry!r}"
        )
        # Permit the optional error pair, reject anything else.
        unexpected = keys - _REQUIRED_FIELDS - _OPTIONAL_ERROR_FIELDS
        assert not unexpected, (
            f"emission #{idx} has unexpected fields {sorted(unexpected)} "
            f"beyond the frozen 11+2 schema: {entry!r}"
        )


def test_kb_ingest_batch_emissions_match_11_field_schema(
    chromadb_service: KnowledgebaseService,
    captured_emissions: list[dict[str, object]],
) -> None:
    """A representative ``kb_ingest_batch`` run emits multiple stderr
    lines; each must comply with the 11-field schema.
    """
    kb = chromadb_service.create_kb("stderr-schema-batch")

    source_defs = [
        {"source_type": "file", "uri": "/tmp/stderr_batch_a.txt"},
        {"source_type": "file", "uri": "/tmp/stderr_batch_b.txt"},
        {"source_type": "file", "uri": "/tmp/stderr_batch_c.txt"},
    ]
    run_ingest_batch(chromadb_service, kb.id, source_defs)

    _assert_schema_compliance(captured_emissions)

    # We expect at minimum: 1 batch_start + 3 per-source + 1 batch_end +
    # lock_acquire/lock_acquired/lock_release for each ingest_source =
    # 1 + 3 + 1 + (3 * 3) = 14 emissions. Don't pin the exact count
    # (refactors may legitimately add lifecycle lines), just bound it.
    ops = [e["op"] for e in captured_emissions]
    assert "ingest_batch_start" in ops
    assert "ingest_batch_end" in ops
    assert ops.count("ingest_source") >= 3


def test_emission_field_types_match_contract(
    chromadb_service: KnowledgebaseService,
    captured_emissions: list[dict[str, object]],
) -> None:
    """Field-type spot-check: ``elapsed_ms`` is int, counters are ints,
    ``request_id`` / ``tool_caller_version`` are str-or-None.
    """
    kb = chromadb_service.create_kb("stderr-schema-types")
    run_ingest_batch(
        chromadb_service,
        kb.id,
        [{"source_type": "file", "uri": "/tmp/types.txt"}],
    )

    for entry in captured_emissions:
        assert isinstance(entry["elapsed_ms"], int), (
            f"elapsed_ms must be int, got {type(entry['elapsed_ms']).__name__}"
        )
        for counter in ("rows_in", "rows_ok", "rows_skipped", "rows_failed"):
            assert isinstance(entry[counter], int), (
                f"{counter} must be int, got {type(entry[counter]).__name__}"
            )
        for nullable_str in ("request_id", "tool_caller_version"):
            value = entry[nullable_str]
            assert value is None or isinstance(value, str), (
                f"{nullable_str} must be str or None, got "
                f"{type(value).__name__}={value!r}"
            )
        for required_str in ("kb_id", "op", "phase", "dedup_policy"):
            assert isinstance(entry[required_str], str), (
                f"{required_str} must be str, got "
                f"{type(entry[required_str]).__name__}"
            )


def test_emission_round_trips_through_json(
    chromadb_service: KnowledgebaseService,
    captured_emissions: list[dict[str, object]],
) -> None:
    """Every payload survives ``json.dumps -> json.loads`` and the result
    matches the original dict (no exotic non-JSON-serialisable values).
    """
    kb = chromadb_service.create_kb("stderr-schema-jsonable")
    run_ingest_batch(
        chromadb_service,
        kb.id,
        [{"source_type": "file", "uri": "/tmp/jsonable.txt"}],
    )

    for entry in captured_emissions:
        encoded = json.dumps(entry)
        decoded = json.loads(encoded)
        assert decoded == entry, (
            f"emission did not round-trip through JSON: {entry!r} "
            f"-> {decoded!r}"
        )
