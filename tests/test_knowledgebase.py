"""Tests for the top-level KnowledgebaseService lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.models import (
    Chunk,
    PageType,
    SourceStatus,
    SourceType,
)
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService
from agent_knowledgebase.services.lint import LintReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embedder() -> MagicMock:
    """A mock Embedder that returns deterministic vectors."""
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    embedder.dimension = 3
    embedder.model_name = "mock-embedder"
    return embedder


@pytest.fixture()
def mock_vectorstore() -> MagicMock:
    """A mock VectorStore."""
    vs = MagicMock()
    vs.add.return_value = None
    vs.delete.return_value = None
    vs.query.return_value = []
    vs.get.return_value = []
    vs.count.return_value = 0
    return vs


@pytest.fixture()
def service(
    test_config: Settings,
    mock_embedder: MagicMock,
    mock_vectorstore: MagicMock,
) -> KnowledgebaseService:
    """Build a KnowledgebaseService with mocked embedder and vectorstore.

    Uses real per-KB SQLite databases (in tmp dirs), a mocked embedder,
    a mocked vectorstore, and a mocked IngestionOrchestrator.
    """
    svc = KnowledgebaseService(test_config)
    # Embedder is lazily created; inject the mock directly so tests don't
    # load sentence-transformers.
    svc._embedder_instance = mock_embedder

    # Override _get_vectorstore to always return the mock
    svc._get_vectorstore = lambda kb_id: mock_vectorstore  # type: ignore[assignment]

    # Mock the ingestion orchestrator to return controlled chunks
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.return_value = [
        Chunk(source_id="", kb_id="", content="chunk one content", metadata={"idx": 0}),
        Chunk(source_id="", kb_id="", content="chunk two content", metadata={"idx": 1}),
    ]
    svc._ingestion = mock_ingestion

    # Make embedder return correct number of embeddings
    def _dynamic_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]] * len(texts)

    mock_embedder.embed.side_effect = _dynamic_embed

    return svc


# ---------------------------------------------------------------------------
# 1. create_kb and get_kb
# ---------------------------------------------------------------------------


class TestCreateAndGetKB:
    def test_create_kb_returns_enriched_kb(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Test KB", "A test knowledgebase")

        assert kb.name == "Test KB"
        assert kb.description == "A test knowledgebase"
        assert kb.id  # UUID was generated
        assert kb.source_count == 0
        assert kb.page_count == 0

    def test_create_kb_creates_directory(self, service: KnowledgebaseService) -> None:
        service.create_kb("Dir Check KB")
        kb_dir = service._config.kb_data_dir("dir-check-kb")
        assert kb_dir.is_dir()
        assert (kb_dir / "knowledgebase.db").exists()

    def test_get_kb_returns_enriched_kb(self, service: KnowledgebaseService) -> None:
        created = service.create_kb("Get Test KB")
        fetched = service.get_kb(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Get Test KB"
        assert fetched.source_count == 0
        assert fetched.page_count == 0

    def test_get_kb_returns_none_for_missing(self, service: KnowledgebaseService) -> None:
        assert service.get_kb("nonexistent-id") is None

    def test_create_kb_with_defaults(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Minimal KB")
        assert kb.description == ""

    def test_create_kb_duplicate_dir_name_raises(self, service: KnowledgebaseService) -> None:
        service.create_kb("My KB")
        with pytest.raises(ValueError, match="already exists"):
            service.create_kb("My KB")


# ---------------------------------------------------------------------------
# 2. list_kbs
# ---------------------------------------------------------------------------


class TestListKBs:
    def test_list_kbs_returns_all(self, service: KnowledgebaseService) -> None:
        service.create_kb("KB One")
        service.create_kb("KB Two")

        kbs = service.list_kbs()
        assert len(kbs) == 2
        names = {kb.name for kb in kbs}
        assert names == {"KB One", "KB Two"}

    def test_list_kbs_empty(self, service: KnowledgebaseService) -> None:
        assert service.list_kbs() == []

    def test_list_kbs_enriched_counts(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Count KB")
        # Ingest a source to add counts
        service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        kbs = service.list_kbs()
        assert len(kbs) == 1
        assert kbs[0].source_count >= 1
        assert kbs[0].page_count >= 1


# ---------------------------------------------------------------------------
# 3. delete_kb cascades
# ---------------------------------------------------------------------------


class TestDeleteKB:
    def test_delete_kb_removes_all_data(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("To Delete")
        service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        # Verify data exists
        assert service.get_kb(kb.id) is not None
        assert len(service.list_sources(kb.id)) >= 1
        assert len(service.list_pages(kb.id)) >= 1

        # Delete
        service.delete_kb(kb.id)

        # Verify cascade
        assert service.get_kb(kb.id) is None

    def test_delete_kb_removes_directory(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Dir Delete KB")
        kb_dir = service._config.kb_data_dir("dir-delete-kb")
        assert kb_dir.is_dir()

        service.delete_kb(kb.id)
        assert not kb_dir.exists()

    def test_delete_nonexistent_kb_does_not_raise(self, service: KnowledgebaseService) -> None:
        """Deleting a missing KB is a no-op (idempotent)."""
        service.delete_kb("does-not-exist")  # should not raise


# ---------------------------------------------------------------------------
# 4. ingest_source runs full pipeline
# ---------------------------------------------------------------------------


class TestIngestSource:
    def test_full_pipeline_produces_source_and_page(
        self,
        service: KnowledgebaseService,
        mock_vectorstore: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        kb = service.create_kb("Ingest KB")
        source = service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        assert source.status == SourceStatus.ingested
        assert source.chunk_count == 2
        assert source.ingested_at is not None
        assert source.kb_id == kb.id

        # Embedder was called with chunk texts
        mock_embedder.embed.assert_called()

        # Vectorstore received the chunks
        mock_vectorstore.add.assert_called_once()
        call_kwargs = mock_vectorstore.add.call_args
        assert len(call_kwargs[1]["ids"]) == 2

        # A summary wiki page was created
        pages = service.list_pages(kb.id)
        assert len(pages) >= 1
        summary_pages = [p for p in pages if p.page_type == PageType.summary]
        assert len(summary_pages) == 1
        assert "Source: /tmp/test.txt" in summary_pages[0].title

    def test_ingest_creates_pipeline_runs(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Pipeline KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        runs = service.get_pipeline_status(kb.id)
        assert len(runs) >= 1

    def test_ingest_nonexistent_kb_raises(self, service: KnowledgebaseService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.ingest_source("no-such-kb", SourceType.file, "/tmp/f.txt")

    def test_ingest_with_metadata(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Meta KB")
        source = service.ingest_source(
            kb.id,
            SourceType.file,
            "/tmp/meta.txt",
            metadata={"author": "test"},
        )
        assert source.metadata == {"author": "test"}

    def test_ingest_failure_marks_source_failed(self, service: KnowledgebaseService) -> None:
        """If ingestion raises, the source should be marked as failed."""
        kb = service.create_kb("Fail KB")

        # Make the ingestion orchestrator raise
        service._ingestion.ingest.side_effect = RuntimeError("read failed")

        with pytest.raises(RuntimeError, match="read failed"):
            service.ingest_source(kb.id, SourceType.file, "/tmp/fail.txt")

        # The source should exist in DB with failed status
        sources = service.list_sources(kb.id)
        assert len(sources) == 1
        assert sources[0].status == SourceStatus.failed

    def test_ingest_empty_chunks(
        self,
        service: KnowledgebaseService,
        mock_vectorstore: MagicMock,
    ) -> None:
        """Ingesting a source that produces no chunks should still succeed."""
        kb = service.create_kb("Empty KB")
        service._ingestion.ingest.return_value = []

        source = service.ingest_source(kb.id, SourceType.file, "/tmp/empty.txt")

        assert source.status == SourceStatus.ingested
        assert source.chunk_count == 0
        # Vectorstore.add should NOT have been called
        mock_vectorstore.add.assert_not_called()


# ---------------------------------------------------------------------------
# 5. remove_source cleans up chunks + vectors
# ---------------------------------------------------------------------------


class TestRemoveSource:
    def test_remove_source_deletes_chunks_and_vectors(
        self,
        service: KnowledgebaseService,
        mock_vectorstore: MagicMock,
    ) -> None:
        kb = service.create_kb("Remove KB")
        source = service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        service.remove_source(source.id)

        # Source record should be gone
        assert service.get_source(source.id) is None
        # Vectorstore.delete should have been called with chunk IDs
        mock_vectorstore.delete.assert_called_once()

    def test_remove_nonexistent_source_raises(self, service: KnowledgebaseService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.remove_source("no-such-source")


# ---------------------------------------------------------------------------
# 6. query / search / hybrid_query delegate to QueryOrchestrator
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_delegates(
        self,
        service: KnowledgebaseService,
        mock_vectorstore: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        kb = service.create_kb("Query KB")
        mock_vectorstore.query.return_value = []

        results = service.query(kb.id, "test question")

        mock_embedder.embed_query.assert_called_with("test question")
        assert isinstance(results, list)

    def test_search_delegates(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Search KB")
        results = service.search(kb.id, "test")
        assert isinstance(results, list)

    def test_hybrid_query_delegates(
        self,
        service: KnowledgebaseService,
        mock_vectorstore: MagicMock,
    ) -> None:
        kb = service.create_kb("Hybrid KB")
        mock_vectorstore.query.return_value = []

        results = service.hybrid_query(kb.id, "test")
        assert isinstance(results, list)

    def test_query_uses_per_kb_embedder_when_dominant_model_differs(
        self,
        service: KnowledgebaseService,
        mock_vectorstore: MagicMock,
        mock_embedder: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the KB's chunks were embedded with a different model than
        the globally configured embedder, query must build an embedder
        matching that model — otherwise retrieval would embed the query
        with the wrong model and produce incoherent scores."""
        kb = service.create_kb("Multi-Model KB")
        # Ingest one source so chunks exist (they'll be stamped with the
        # mock embedder's model name "mock-embedder").
        service.ingest_source(kb.id, SourceType.file, "/tmp/x.txt")

        # Simulate the stored vectorstore having been built with a
        # *different* embedding model than the service's current default
        # by patching the DB-side count helper.
        ctx = service._contexts[kb.id]
        monkeypatch.setattr(
            ctx.db,
            "count_chunks_by_embedding_model",
            lambda _kb_id: {"legacy-model": 7},
        )

        captured: dict[str, str] = {}

        def _fake_builder(cfg: object, model_name: str | None) -> MagicMock:
            captured["model"] = model_name or ""
            new_embedder = MagicMock()
            new_embedder.embed_query.return_value = [0.9, 0.9, 0.9]
            return new_embedder

        monkeypatch.setattr(
            "agent_knowledgebase.services.knowledgebase.create_embedder_for_model",
            _fake_builder,
        )
        mock_vectorstore.query.return_value = []

        service.query(kb.id, "test question")

        assert captured["model"] == "legacy-model"

    def test_query_reuses_default_embedder_when_model_matches(
        self,
        service: KnowledgebaseService,
        mock_vectorstore: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        """When the KB's stored embedding model matches the service's
        globally configured embedder, query must reuse that embedder
        (no rebuild) to avoid repeated sentence-transformers model loads."""
        kb = service.create_kb("Same-Model KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/x.txt")

        mock_vectorstore.query.return_value = []
        service.query(kb.id, "hello")

        mock_embedder.embed_query.assert_called_with("hello")


# ---------------------------------------------------------------------------
# 7. Wiki operations
# ---------------------------------------------------------------------------


class TestWikiOperations:
    def test_get_page(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Wiki KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        pages = service.list_pages(kb.id)
        assert len(pages) >= 1

        page = service.get_page(pages[0].id)
        assert page is not None
        assert page.id == pages[0].id

    def test_get_page_missing(self, service: KnowledgebaseService) -> None:
        assert service.get_page("nonexistent") is None

    def test_list_pages_by_type(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Type KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        summary_pages = service.list_pages(kb.id, page_type=PageType.summary)
        all_pages = service.list_pages(kb.id)

        assert len(summary_pages) <= len(all_pages)
        assert all(p.page_type == PageType.summary for p in summary_pages)

    def test_list_sources(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Sources KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/a.txt")

        sources = service.list_sources(kb.id)
        assert len(sources) >= 1
        assert sources[0].kb_id == kb.id

    def test_get_source(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Get Source KB")
        source = service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        fetched = service.get_source(source.id)
        assert fetched is not None
        assert fetched.id == source.id

    def test_get_source_missing(self, service: KnowledgebaseService) -> None:
        assert service.get_source("nonexistent") is None


# ---------------------------------------------------------------------------
# 8. lint and lint_fix delegate correctly
# ---------------------------------------------------------------------------


class TestLint:
    def test_lint_returns_report(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Lint KB")
        report = service.lint(kb.id)

        assert isinstance(report, LintReport)
        assert isinstance(report.issues, list)
        assert report.checked_at is not None

    def test_lint_fix_returns_fixed_issues(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Lint Fix KB")
        fixed = service.lint_fix(kb.id)

        assert isinstance(fixed, list)


# ---------------------------------------------------------------------------
# 9. rebuild_index
# ---------------------------------------------------------------------------


class TestRebuildIndex:
    def test_rebuild_creates_index_page(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Index KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        index_page = service.rebuild_index(kb.id)
        assert index_page.page_type == PageType.index
        assert index_page.title == "Index"
        assert "Summary" in index_page.content


# ---------------------------------------------------------------------------
# 10. export delegates to MarkdownExporter
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_creates_files(self, service: KnowledgebaseService, tmp_path: Path) -> None:
        kb = service.create_kb("Export KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        output_dir = tmp_path / "export_output"
        paths = service.export(kb.id, output_dir)

        assert len(paths) >= 1
        assert all(p.exists() for p in paths)
        assert all(p.suffix == ".md" for p in paths)

    def test_export_empty_kb(self, service: KnowledgebaseService, tmp_path: Path) -> None:
        kb = service.create_kb("Empty Export KB")
        output_dir = tmp_path / "empty_export"
        paths = service.export(kb.id, output_dir)
        assert paths == []


# ---------------------------------------------------------------------------
# 11. get_pipeline_status
# ---------------------------------------------------------------------------


class TestPipelineStatus:
    def test_returns_runs_for_kb(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Pipeline Status KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/test.txt")

        runs = service.get_pipeline_status(kb.id)
        assert len(runs) >= 1
        assert all(r.kb_id == kb.id for r in runs)

    def test_empty_for_new_kb(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("New KB")
        assert service.get_pipeline_status(kb.id) == []


# ---------------------------------------------------------------------------
# 12. _enrich_kb correctness
# ---------------------------------------------------------------------------


class TestEnrichKB:
    def test_counts_reflect_ingested_data(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Enrich KB")
        service.ingest_source(kb.id, SourceType.file, "/tmp/a.txt")

        enriched = service.get_kb(kb.id)
        assert enriched is not None
        assert enriched.source_count >= 1
        assert enriched.page_count >= 1

    def test_counts_zero_for_empty_kb(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Empty Enrich KB")
        enriched = service.get_kb(kb.id)
        assert enriched is not None
        assert enriched.source_count == 0
        assert enriched.page_count == 0


# ---------------------------------------------------------------------------
# 13. Per-KB directory isolation
# ---------------------------------------------------------------------------


class TestPerKBIsolation:
    def test_two_kbs_have_separate_directories(self, service: KnowledgebaseService) -> None:
        service.create_kb("Alpha KB")
        service.create_kb("Beta KB")

        dir1 = service._config.kb_data_dir("alpha-kb")
        dir2 = service._config.kb_data_dir("beta-kb")

        assert dir1.is_dir()
        assert dir2.is_dir()
        assert dir1 != dir2

    def test_deleting_one_kb_preserves_others(self, service: KnowledgebaseService) -> None:
        kb1 = service.create_kb("Keep KB")
        kb2 = service.create_kb("Remove KB")
        service.ingest_source(kb1.id, SourceType.file, "/tmp/a.txt")

        service.delete_kb(kb2.id)

        # kb1 should still be fully intact
        assert service.get_kb(kb1.id) is not None
        assert len(service.list_sources(kb1.id)) >= 1

    def test_index_file_persisted(self, service: KnowledgebaseService) -> None:
        kb = service.create_kb("Index File KB")
        index_path = service._base_dir / ".index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert kb.id in data
