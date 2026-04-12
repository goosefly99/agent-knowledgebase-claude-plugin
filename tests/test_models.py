"""Tests for agent_knowledgebase.models."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from agent_knowledgebase.models import (
    Chunk,
    Knowledgebase,
    PageType,
    PipelinePhase,
    PipelineRun,
    RunStatus,
    Source,
    SourceStatus,
    SourceType,
    WikiPage,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_source_type_values(self) -> None:
        assert set(SourceType) == {
            SourceType.file,
            SourceType.directory,
            SourceType.codebase,
            SourceType.website,
            SourceType.sql_database,
            SourceType.git_history,
            SourceType.api_endpoint,
        }

    def test_source_status_values(self) -> None:
        assert set(SourceStatus) == {
            SourceStatus.pending,
            SourceStatus.ingesting,
            SourceStatus.ingested,
            SourceStatus.failed,
        }

    def test_page_type_values(self) -> None:
        assert set(PageType) == {
            PageType.entity,
            PageType.concept,
            PageType.summary,
            PageType.index,
            PageType.comparison,
            PageType.synthesis,
        }

    def test_pipeline_phase_values(self) -> None:
        assert set(PipelinePhase) == {
            PipelinePhase.initialize,
            PipelinePhase.read_source,
            PipelinePhase.chunk,
            PipelinePhase.embed,
            PipelinePhase.integrate_wiki,
            PipelinePhase.finalize,
        }

    def test_run_status_values(self) -> None:
        assert set(RunStatus) == {
            RunStatus.pending,
            RunStatus.running,
            RunStatus.completed,
            RunStatus.failed,
        }

    def test_enums_are_str(self) -> None:
        """All enums should behave as strings for JSON serialisation."""
        assert SourceType.file == "file"
        assert SourceStatus.pending == "pending"
        assert PageType.entity == "entity"
        assert PipelinePhase.chunk == "chunk"
        assert RunStatus.running == "running"


# ---------------------------------------------------------------------------
# Knowledgebase tests
# ---------------------------------------------------------------------------


class TestKnowledgebase:
    def test_defaults(self) -> None:
        kb = Knowledgebase(name="test-kb")
        assert kb.name == "test-kb"
        assert kb.description == ""
        assert isinstance(kb.id, str) and len(kb.id) > 0
        assert isinstance(kb.created_at, datetime)
        assert isinstance(kb.updated_at, datetime)
        assert kb.source_count == 0
        assert kb.page_count == 0
        assert kb.config == {}

    def test_custom_fields(self) -> None:
        kb = Knowledgebase(
            name="my-kb",
            description="A test knowledgebase",
            config={"key": "value"},
        )
        assert kb.description == "A test knowledgebase"
        assert kb.config == {"key": "value"}

    def test_unique_ids(self) -> None:
        kb1 = Knowledgebase(name="a")
        kb2 = Knowledgebase(name="b")
        assert kb1.id != kb2.id

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            Knowledgebase()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Source tests
# ---------------------------------------------------------------------------


class TestSource:
    def test_defaults(self) -> None:
        src = Source(kb_id="kb-1", source_type=SourceType.file, uri="/tmp/x.txt")
        assert src.status == SourceStatus.pending
        assert src.chunk_count == 0
        assert src.metadata == {}
        assert src.ingested_at is None

    def test_source_type_from_string(self) -> None:
        src = Source(kb_id="kb-1", source_type="website", uri="https://example.com")
        assert src.source_type is SourceType.website

    def test_invalid_source_type(self) -> None:
        with pytest.raises(ValidationError):
            Source(kb_id="kb-1", source_type="foobar", uri="/tmp/x.txt")

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Source()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# WikiPage tests
# ---------------------------------------------------------------------------


class TestWikiPage:
    def test_defaults(self) -> None:
        page = WikiPage(
            kb_id="kb-1",
            title="Test Page",
            content="# Hello",
            page_type=PageType.entity,
        )
        assert page.tags == []
        assert page.source_ids == []
        assert page.inbound_links == []
        assert page.outbound_links == []
        assert isinstance(page.created_at, datetime)

    def test_with_links_and_tags(self) -> None:
        page = WikiPage(
            kb_id="kb-1",
            title="Linked",
            content="body",
            page_type=PageType.concept,
            tags=["a", "b"],
            source_ids=["s1"],
            outbound_links=["p2"],
            inbound_links=["p0"],
        )
        assert page.tags == ["a", "b"]
        assert page.outbound_links == ["p2"]
        assert page.inbound_links == ["p0"]

    def test_page_type_from_string(self) -> None:
        page = WikiPage(
            kb_id="kb-1", title="T", content="C", page_type="summary"
        )
        assert page.page_type is PageType.summary


# ---------------------------------------------------------------------------
# Chunk tests
# ---------------------------------------------------------------------------


class TestChunk:
    def test_defaults(self) -> None:
        chunk = Chunk(source_id="s-1", kb_id="kb-1", content="hello world")
        assert chunk.metadata == {}
        assert chunk.embedding_id is None
        assert isinstance(chunk.id, str)

    def test_with_embedding(self) -> None:
        chunk = Chunk(
            source_id="s-1",
            kb_id="kb-1",
            content="text",
            embedding_id="emb-123",
        )
        assert chunk.embedding_id == "emb-123"


# ---------------------------------------------------------------------------
# PipelineRun tests
# ---------------------------------------------------------------------------


class TestPipelineRun:
    def test_defaults(self) -> None:
        run = PipelineRun(
            kb_id="kb-1",
            phase=PipelinePhase.initialize,
        )
        assert run.status == RunStatus.pending
        assert run.source_id is None
        assert run.completed_at is None
        assert run.error is None
        assert run.metadata == {}
        assert isinstance(run.started_at, datetime)

    def test_failed_run(self) -> None:
        run = PipelineRun(
            kb_id="kb-1",
            phase=PipelinePhase.embed,
            status=RunStatus.failed,
            error="embedding service unavailable",
        )
        assert run.status is RunStatus.failed
        assert run.error == "embedding service unavailable"

    def test_serialisation_roundtrip(self) -> None:
        run = PipelineRun(
            kb_id="kb-1",
            phase=PipelinePhase.chunk,
            metadata={"chunks": 42},
        )
        data = run.model_dump()
        restored = PipelineRun(**data)
        assert restored.id == run.id
        assert restored.metadata == {"chunks": 42}
