"""Tests for agent_knowledgebase.database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent_knowledgebase.database import Database
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
# Schema and initialisation
# ---------------------------------------------------------------------------


class TestSchemaInit:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "test.db"
        db = Database(db_path=deep)
        assert deep.exists()
        db.close()

    def test_wal_mode(self, test_db: Database) -> None:
        mode = test_db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, test_db: Database) -> None:
        fk = test_db._conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_tables_exist(self, test_db: Database) -> None:
        rows = test_db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r["name"] for r in rows}
        expected = {
            "knowledgebases",
            "sources",
            "wiki_pages",
            "wiki_links",
            "page_sources",
            "chunks",
            "pipeline_runs",
            "wiki_pages_fts",
            # FTS shadow tables are also present but we only check the main ones.
        }
        assert expected.issubset(names)

    def test_idempotent_init(self, tmp_path: Path) -> None:
        """Opening the same DB twice must not raise."""
        path = tmp_path / "idem.db"
        db1 = Database(db_path=path)
        db1.close()
        db2 = Database(db_path=path)
        db2.close()


# ---------------------------------------------------------------------------
# Knowledgebase CRUD
# ---------------------------------------------------------------------------


class TestKnowledgebaseCRUD:
    def test_insert_and_get(self, test_db: Database) -> None:
        kb = Knowledgebase(name="my-kb", description="test", config={"k": 1})
        test_db.insert_knowledgebase(kb)
        fetched = test_db.get_knowledgebase(kb.id)
        assert fetched is not None
        assert fetched.name == "my-kb"
        assert fetched.description == "test"
        assert fetched.config == {"k": 1}

    def test_get_missing(self, test_db: Database) -> None:
        assert test_db.get_knowledgebase("nonexistent") is None

    def test_list(self, test_db: Database) -> None:
        test_db.insert_knowledgebase(Knowledgebase(name="a"))
        test_db.insert_knowledgebase(Knowledgebase(name="b"))
        kbs = test_db.list_knowledgebases()
        assert len(kbs) == 2

    def test_delete(self, test_db: Database) -> None:
        kb = Knowledgebase(name="to-delete")
        test_db.insert_knowledgebase(kb)
        test_db.delete_knowledgebase(kb.id)
        assert test_db.get_knowledgebase(kb.id) is None

    def test_unique_name_constraint(self, test_db: Database) -> None:
        test_db.insert_knowledgebase(Knowledgebase(name="dup"))
        with pytest.raises(sqlite3.IntegrityError):
            test_db.insert_knowledgebase(Knowledgebase(name="dup"))


# ---------------------------------------------------------------------------
# Source CRUD
# ---------------------------------------------------------------------------


class TestSourceCRUD:
    @pytest.fixture(autouse=True)
    def _kb(self, test_db: Database) -> None:
        self.kb = Knowledgebase(name="src-kb")
        test_db.insert_knowledgebase(self.kb)

    def test_insert_and_get(self, test_db: Database) -> None:
        src = Source(
            kb_id=self.kb.id,
            source_type=SourceType.file,
            uri="/tmp/f.txt",
            metadata={"size": 100},
        )
        test_db.insert_source(src)
        fetched = test_db.get_source(src.id)
        assert fetched is not None
        assert fetched.uri == "/tmp/f.txt"
        assert fetched.source_type is SourceType.file
        assert fetched.metadata == {"size": 100}
        assert fetched.status is SourceStatus.pending

    def test_list(self, test_db: Database) -> None:
        for i in range(3):
            test_db.insert_source(
                Source(kb_id=self.kb.id, source_type=SourceType.website, uri=f"https://{i}")
            )
        assert len(test_db.list_sources(self.kb.id)) == 3

    def test_delete(self, test_db: Database) -> None:
        src = Source(kb_id=self.kb.id, source_type=SourceType.directory, uri="/tmp/d")
        test_db.insert_source(src)
        test_db.delete_source(src.id)
        assert test_db.get_source(src.id) is None


# ---------------------------------------------------------------------------
# WikiPage CRUD
# ---------------------------------------------------------------------------


class TestWikiPageCRUD:
    @pytest.fixture(autouse=True)
    def _kb(self, test_db: Database) -> None:
        self.kb = Knowledgebase(name="wiki-kb")
        test_db.insert_knowledgebase(self.kb)

    def test_insert_and_get(self, test_db: Database) -> None:
        page = WikiPage(
            kb_id=self.kb.id,
            title="Overview",
            content="# Overview\nHello.",
            page_type=PageType.summary,
            tags=["intro"],
        )
        test_db.insert_wiki_page(page)
        fetched = test_db.get_wiki_page(page.id)
        assert fetched is not None
        assert fetched.title == "Overview"
        assert fetched.tags == ["intro"]
        assert fetched.page_type is PageType.summary

    def test_list(self, test_db: Database) -> None:
        for i in range(2):
            test_db.insert_wiki_page(
                WikiPage(
                    kb_id=self.kb.id,
                    title=f"Page {i}",
                    content="body",
                    page_type=PageType.entity,
                )
            )
        assert len(test_db.list_wiki_pages(self.kb.id)) == 2

    def test_delete(self, test_db: Database) -> None:
        page = WikiPage(
            kb_id=self.kb.id, title="Gone", content="x", page_type=PageType.entity
        )
        test_db.insert_wiki_page(page)
        test_db.delete_wiki_page(page.id)
        assert test_db.get_wiki_page(page.id) is None

    def test_source_ids_persisted(self, test_db: Database) -> None:
        src = Source(kb_id=self.kb.id, source_type=SourceType.file, uri="/f")
        test_db.insert_source(src)
        page = WikiPage(
            kb_id=self.kb.id,
            title="Linked",
            content="c",
            page_type=PageType.entity,
            source_ids=[src.id],
        )
        test_db.insert_wiki_page(page)
        fetched = test_db.get_wiki_page(page.id)
        assert fetched is not None
        assert src.id in fetched.source_ids

    def test_outbound_links_persisted(self, test_db: Database) -> None:
        p1 = WikiPage(
            kb_id=self.kb.id, title="A", content="a", page_type=PageType.entity
        )
        p2 = WikiPage(
            kb_id=self.kb.id,
            title="B",
            content="b",
            page_type=PageType.entity,
        )
        test_db.insert_wiki_page(p1)
        # p2 links to p1 on creation.
        p2_linked = WikiPage(
            id=p2.id,
            kb_id=self.kb.id,
            title="B",
            content="b",
            page_type=PageType.entity,
            outbound_links=[p1.id],
        )
        test_db.insert_wiki_page(p2_linked)
        fetched = test_db.get_wiki_page(p2.id)
        assert fetched is not None
        assert p1.id in fetched.outbound_links
        # p1 should see an inbound link from p2.
        fetched_p1 = test_db.get_wiki_page(p1.id)
        assert fetched_p1 is not None
        assert p2.id in fetched_p1.inbound_links


# ---------------------------------------------------------------------------
# Wiki links
# ---------------------------------------------------------------------------


class TestWikiLinks:
    @pytest.fixture(autouse=True)
    def _pages(self, test_db: Database) -> None:
        self.kb = Knowledgebase(name="link-kb")
        test_db.insert_knowledgebase(self.kb)
        self.p1 = WikiPage(
            kb_id=self.kb.id, title="P1", content="p1", page_type=PageType.entity
        )
        self.p2 = WikiPage(
            kb_id=self.kb.id, title="P2", content="p2", page_type=PageType.entity
        )
        test_db.insert_wiki_page(self.p1)
        test_db.insert_wiki_page(self.p2)

    def test_add_and_get(self, test_db: Database) -> None:
        test_db.add_wiki_link(self.p1.id, self.p2.id)
        assert test_db.get_wiki_links(self.p1.id, "outbound") == [self.p2.id]
        assert test_db.get_wiki_links(self.p2.id, "inbound") == [self.p1.id]

    def test_remove(self, test_db: Database) -> None:
        test_db.add_wiki_link(self.p1.id, self.p2.id)
        test_db.remove_wiki_link(self.p1.id, self.p2.id)
        assert test_db.get_wiki_links(self.p1.id, "outbound") == []

    def test_duplicate_ignored(self, test_db: Database) -> None:
        test_db.add_wiki_link(self.p1.id, self.p2.id)
        test_db.add_wiki_link(self.p1.id, self.p2.id)  # should not raise
        assert test_db.get_wiki_links(self.p1.id, "outbound") == [self.p2.id]


# ---------------------------------------------------------------------------
# Page-source associations
# ---------------------------------------------------------------------------


class TestPageSources:
    def test_add_and_get(self, test_db: Database) -> None:
        kb = Knowledgebase(name="ps-kb")
        test_db.insert_knowledgebase(kb)
        src = Source(kb_id=kb.id, source_type=SourceType.file, uri="/f")
        test_db.insert_source(src)
        page = WikiPage(
            kb_id=kb.id, title="PS", content="c", page_type=PageType.entity
        )
        test_db.insert_wiki_page(page)
        test_db.add_page_source(page.id, src.id)
        assert test_db.get_page_sources(page.id) == [src.id]


# ---------------------------------------------------------------------------
# Chunk CRUD
# ---------------------------------------------------------------------------


class TestChunkCRUD:
    @pytest.fixture(autouse=True)
    def _setup(self, test_db: Database) -> None:
        self.kb = Knowledgebase(name="chunk-kb")
        test_db.insert_knowledgebase(self.kb)
        self.src = Source(
            kb_id=self.kb.id, source_type=SourceType.file, uri="/tmp/c.txt"
        )
        test_db.insert_source(self.src)

    def test_insert_and_get(self, test_db: Database) -> None:
        chunk = Chunk(
            source_id=self.src.id,
            kb_id=self.kb.id,
            content="hello",
            metadata={"pos": 0},
            embedding_id="emb-1",
        )
        test_db.insert_chunk(chunk)
        fetched = test_db.get_chunk(chunk.id)
        assert fetched is not None
        assert fetched.content == "hello"
        assert fetched.metadata == {"pos": 0}
        assert fetched.embedding_id == "emb-1"

    def test_list(self, test_db: Database) -> None:
        for i in range(3):
            test_db.insert_chunk(
                Chunk(source_id=self.src.id, kb_id=self.kb.id, content=f"chunk {i}")
            )
        assert len(test_db.list_chunks(self.src.id)) == 3

    def test_delete(self, test_db: Database) -> None:
        chunk = Chunk(source_id=self.src.id, kb_id=self.kb.id, content="bye")
        test_db.insert_chunk(chunk)
        test_db.delete_chunk(chunk.id)
        assert test_db.get_chunk(chunk.id) is None


# ---------------------------------------------------------------------------
# PipelineRun CRUD
# ---------------------------------------------------------------------------


class TestPipelineRunCRUD:
    @pytest.fixture(autouse=True)
    def _setup(self, test_db: Database) -> None:
        self.kb = Knowledgebase(name="run-kb")
        test_db.insert_knowledgebase(self.kb)

    def test_insert_and_get(self, test_db: Database) -> None:
        run = PipelineRun(
            kb_id=self.kb.id,
            phase=PipelinePhase.embed,
            status=RunStatus.running,
            metadata={"chunks": 10},
        )
        test_db.insert_pipeline_run(run)
        fetched = test_db.get_pipeline_run(run.id)
        assert fetched is not None
        assert fetched.phase is PipelinePhase.embed
        assert fetched.status is RunStatus.running
        assert fetched.metadata == {"chunks": 10}

    def test_list(self, test_db: Database) -> None:
        for phase in (PipelinePhase.initialize, PipelinePhase.chunk):
            test_db.insert_pipeline_run(
                PipelineRun(kb_id=self.kb.id, phase=phase)
            )
        assert len(test_db.list_pipeline_runs(self.kb.id)) == 2

    def test_delete(self, test_db: Database) -> None:
        run = PipelineRun(kb_id=self.kb.id, phase=PipelinePhase.finalize)
        test_db.insert_pipeline_run(run)
        test_db.delete_pipeline_run(run.id)
        assert test_db.get_pipeline_run(run.id) is None

    def test_failed_run_with_error(self, test_db: Database) -> None:
        run = PipelineRun(
            kb_id=self.kb.id,
            phase=PipelinePhase.read_source,
            status=RunStatus.failed,
            error="Connection refused",
        )
        test_db.insert_pipeline_run(run)
        fetched = test_db.get_pipeline_run(run.id)
        assert fetched is not None
        assert fetched.error == "Connection refused"


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------


class TestFTSSearch:
    @pytest.fixture(autouse=True)
    def _setup(self, test_db: Database) -> None:
        self.kb = Knowledgebase(name="fts-kb")
        test_db.insert_knowledgebase(self.kb)
        self.page = WikiPage(
            kb_id=self.kb.id,
            title="Python Guide",
            content="Python is a versatile programming language used in many domains.",
            page_type=PageType.summary,
            tags=["python", "guide"],
        )
        test_db.insert_wiki_page(self.page)

    def test_search_by_title(self, test_db: Database) -> None:
        results = test_db.search_wiki_fts("Python", self.kb.id)
        assert len(results) >= 1
        assert any(r.id == self.page.id for r in results)

    def test_search_by_content(self, test_db: Database) -> None:
        results = test_db.search_wiki_fts("versatile programming", self.kb.id)
        assert len(results) >= 1

    def test_search_no_match(self, test_db: Database) -> None:
        results = test_db.search_wiki_fts("nonexistentxyz", self.kb.id)
        assert results == []

    def test_search_scoped_to_kb(self, test_db: Database) -> None:
        other_kb = Knowledgebase(name="other-kb")
        test_db.insert_knowledgebase(other_kb)
        results = test_db.search_wiki_fts("Python", other_kb.id)
        assert results == []

    def test_fts_after_delete(self, test_db: Database) -> None:
        results = test_db.search_wiki_fts("Python", self.kb.id)
        assert len(results) >= 1
        test_db.delete_wiki_page(self.page.id)
        results = test_db.search_wiki_fts("Python", self.kb.id)
        assert results == []
