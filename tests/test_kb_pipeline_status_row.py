"""Tests for the v0.6.0 ``kb_pipeline_status`` telemetry row.

Covers the schema migration (additive + idempotent), the new
:class:`PipelineRun` pydantic fields, end-to-end row population under
every dedup policy, and the stability invariant that
``server.kb_pipeline_status`` keeps the exact signature
``(kb_id: str) -> str`` (enforced via AST parsing so the assertion
doesn't depend on runtime import state).
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.database import Database
from agent_knowledgebase.models import Chunk, PipelineRun, SourceType
from agent_knowledgebase.services.dedup_service import DedupPolicy
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# ---------------------------------------------------------------------------
# PipelineRun model surface
# ---------------------------------------------------------------------------


_NEW_FIELDS = [
    "ended_at",
    "ingested",
    "skipped",
    "replaced",
    "failed",
    "batch_size",
    "dedup_policy",
    "request_id",
    "tool_caller_version",
]


def test_pipeline_run_model_has_new_fields() -> None:
    """All 9 new telemetry fields are present, optional, and default to None."""
    fields = PipelineRun.model_fields
    for name in _NEW_FIELDS:
        assert name in fields, f"PipelineRun missing new field {name!r}"
        info = fields[name]
        # Default is None (Optional additive).
        assert info.default is None, (
            f"PipelineRun.{name} default must be None (got {info.default!r})"
        )


# ---------------------------------------------------------------------------
# DDL migration
# ---------------------------------------------------------------------------


def _table_columns(db_path: Path, table: str) -> dict[str, str]:
    """Return a ``{column_name: declared_type}`` snapshot for *table*."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1]: row[2] for row in rows}
    finally:
        conn.close()


_EXPECTED_TYPES: dict[str, str] = {
    "ended_at": "TEXT",
    "ingested": "INTEGER",
    "skipped": "INTEGER",
    "replaced": "INTEGER",
    "failed": "INTEGER",
    "batch_size": "INTEGER",
    "dedup_policy": "TEXT",
    "request_id": "TEXT",
    "tool_caller_version": "TEXT",
}


def test_database_schema_has_new_columns_after_init(tmp_path: Path) -> None:
    """After ``Database.__init__``, pipeline_runs has all 9 new columns."""
    db = Database(db_path=tmp_path / "kb.db")
    try:
        cols = _table_columns(tmp_path / "kb.db", "pipeline_runs")
        for name, expected_type in _EXPECTED_TYPES.items():
            assert name in cols, f"pipeline_runs missing column {name!r}"
            assert cols[name].upper() == expected_type, (
                f"pipeline_runs.{name} type {cols[name]!r} != {expected_type!r}"
            )
    finally:
        db.close()


def test_migration_idempotent_second_init_is_noop(tmp_path: Path) -> None:
    """Re-opening the DB triggers ``_init_schema`` again with zero effect."""
    db_path = tmp_path / "kb.db"
    first = Database(db_path=db_path)
    try:
        cols_first = _table_columns(db_path, "pipeline_runs")
    finally:
        first.close()
    second = Database(db_path=db_path)
    try:
        cols_second = _table_columns(db_path, "pipeline_runs")
    finally:
        second.close()
    assert cols_first == cols_second, (
        "Column set must be unchanged across consecutive _init_schema invocations"
    )


# ---------------------------------------------------------------------------
# End-to-end: telemetry row populated per dedup policy
# ---------------------------------------------------------------------------


@pytest.fixture()
def telemetry_service(tmp_path: Path) -> KnowledgebaseService:
    """KnowledgebaseService with mocked embedder, vectorstore, and ingestion."""
    saves = tmp_path / "saves"
    saves.mkdir()
    config = Settings(saves_dir=saves, export_path=tmp_path / "export")
    svc = KnowledgebaseService(config)
    svc._embedder_instance = MagicMock()
    svc._embedder_instance.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    svc._embedder_instance.model_name = "test-embedder"
    svc._get_vectorstore = lambda kb_id: MagicMock(  # type: ignore[assignment]
        add=MagicMock(return_value=None),
        delete=MagicMock(return_value=None),
    )
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.side_effect = lambda *_, **__: [
        Chunk(source_id="", kb_id="", content="hello world", metadata={})
    ]
    svc._ingestion = mock_ingestion
    return svc


def _latest_run_for_policy(svc: KnowledgebaseService, kb_id: str) -> PipelineRun:
    """Return the most-recently-inserted PipelineRun row for *kb_id*."""
    ctx = svc._ctx(kb_id)
    runs = ctx.db.list_pipeline_runs(kb_id)
    assert runs, f"no pipeline runs recorded for kb_id={kb_id!r}"
    return runs[-1]


@pytest.mark.parametrize(
    "policy, expected_ingested, expected_skipped, expected_replaced",
    [
        (DedupPolicy.skip, 0, 1, 0),
        (DedupPolicy.replace, 1, 0, 1),
        (DedupPolicy.force_add, 1, 0, 0),
    ],
)
def test_pipeline_run_row_populated_on_ingest_each_dedup_policy(
    telemetry_service: KnowledgebaseService,
    policy: DedupPolicy,
    expected_ingested: int,
    expected_skipped: int,
    expected_replaced: int,
) -> None:
    """Each dedup policy records the expected counter pattern on the run row."""
    kb = telemetry_service.create_kb(f"kb-{policy.value}")

    # First ingest establishes the dedup_key under skip (baseline).
    telemetry_service.ingest_source(
        kb.id,
        SourceType.file,
        "/tmp/a.txt",
        dedup_key="X",
        dedup_policy=DedupPolicy.skip,
        request_id="req-1",
        tool_caller_version="0.6.0",
        batch_size=2,
    )
    # Second ingest exercises the target policy.
    telemetry_service.ingest_source(
        kb.id,
        SourceType.file,
        "/tmp/b.txt",
        dedup_key="X",
        dedup_policy=policy,
        request_id="req-2",
        tool_caller_version="0.6.0",
        batch_size=2,
    )

    latest_run = _latest_run_for_policy(telemetry_service, kb.id)

    # All 9 v0.6.0 fields must be populated (non-None).
    assert latest_run.ended_at is not None
    assert latest_run.ingested is not None
    assert latest_run.skipped is not None
    assert latest_run.replaced is not None
    assert latest_run.failed is not None
    assert latest_run.batch_size is not None
    assert latest_run.dedup_policy is not None
    assert latest_run.request_id is not None
    assert latest_run.tool_caller_version is not None

    # Counter values must match the policy semantics.
    assert latest_run.ingested == expected_ingested, (
        f"ingested: {latest_run.ingested} != {expected_ingested} for {policy.value}"
    )
    assert latest_run.skipped == expected_skipped, (
        f"skipped: {latest_run.skipped} != {expected_skipped} for {policy.value}"
    )
    assert latest_run.replaced == expected_replaced, (
        f"replaced: {latest_run.replaced} != {expected_replaced} for {policy.value}"
    )
    assert latest_run.failed == 0
    assert latest_run.batch_size == 2
    assert latest_run.dedup_policy == policy.value
    assert latest_run.request_id == "req-2"
    assert latest_run.tool_caller_version == "0.6.0"


# ---------------------------------------------------------------------------
# Invariant: kb_pipeline_status MCP tool signature is unchanged.
# ---------------------------------------------------------------------------


def _find_function_def(module_ast: ast.Module, name: str) -> ast.FunctionDef:
    """Return the top-level ``FunctionDef`` node with *name*."""
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"FunctionDef {name!r} not found")


def test_kb_pipeline_status_tool_signature_unchanged() -> None:
    """The MCP tool's function signature is exactly ``(kb_id: str) -> str``.

    Asserted via AST parsing of ``src/agent_knowledgebase/server.py`` so
    the check is independent of any import-time state. This is the
    SC5 release gate: adding or renaming parameters on this tool is a
    breaking change for MCP callers.
    """
    import agent_knowledgebase.server as _server_module

    source_path = Path(_server_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    func = _find_function_def(tree, "kb_pipeline_status")

    # Positional args == ['kb_id'] — no kwonly, no varargs, no kwargs.
    positional_names = [a.arg for a in func.args.args]
    assert positional_names == ["kb_id"], (
        f"kb_pipeline_status args changed: {positional_names!r} != ['kb_id']"
    )
    assert func.args.vararg is None
    assert func.args.kwarg is None
    assert func.args.kwonlyargs == []
    assert func.args.defaults == []
    # Annotation of kb_id must be 'str'.
    kb_id_annotation = func.args.args[0].annotation
    assert isinstance(kb_id_annotation, ast.Name)
    assert kb_id_annotation.id == "str"
    # Return annotation must be 'str'.
    assert func.returns is not None
    assert isinstance(func.returns, ast.Name)
    assert func.returns.id == "str"


# ---------------------------------------------------------------------------
# Ensure the row survives a round-trip through the DB layer.
# ---------------------------------------------------------------------------


def test_pipeline_run_round_trips_new_columns_through_database(tmp_path: Path) -> None:
    """insert_pipeline_run + _row_to_pipeline_run preserve the 9 new fields."""
    from datetime import UTC, datetime

    from agent_knowledgebase.models import Knowledgebase, PipelinePhase, RunStatus

    db = Database(db_path=tmp_path / "kb.db")
    try:
        kb = Knowledgebase(name="round-trip-kb")
        db.insert_knowledgebase(kb)
        now = datetime.now(UTC)
        run = PipelineRun(
            kb_id=kb.id,
            phase=PipelinePhase.initialize,
            status=RunStatus.completed,
            ended_at=now,
            ingested=1,
            skipped=0,
            replaced=0,
            failed=0,
            batch_size=3,
            dedup_policy="skip",
            request_id="req-rt",
            tool_caller_version="0.6.0",
        )
        db.insert_pipeline_run(run)
        runs = db.list_pipeline_runs(kb.id)
        assert len(runs) == 1
        loaded = runs[0]
        assert loaded.ended_at is not None
        assert loaded.ingested == 1
        assert loaded.skipped == 0
        assert loaded.replaced == 0
        assert loaded.failed == 0
        assert loaded.batch_size == 3
        assert loaded.dedup_policy == "skip"
        assert loaded.request_id == "req-rt"
        assert loaded.tool_caller_version == "0.6.0"
    finally:
        db.close()
