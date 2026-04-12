"""Tests for agent_knowledgebase.services.export (MarkdownExporter)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_knowledgebase.database import Database
from agent_knowledgebase.models import Knowledgebase, PageType, Source, SourceType
from agent_knowledgebase.services.export import MarkdownExporter, _sanitize_filename
from agent_knowledgebase.services.wiki import WikiManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def export_kb(test_db: Database) -> Knowledgebase:
    """Insert and return a knowledgebase for export tests."""
    kb = Knowledgebase(name="export-test-kb")
    test_db.insert_knowledgebase(kb)
    return kb


@pytest.fixture()
def wiki(test_db: Database) -> WikiManager:
    return WikiManager(test_db)


@pytest.fixture()
def exporter(wiki: WikiManager) -> MarkdownExporter:
    return MarkdownExporter(wiki)


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "export_output"


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_basic(self) -> None:
        assert _sanitize_filename("My Page Title") == "my-page-title"

    def test_special_chars(self) -> None:
        assert _sanitize_filename("Page: A/B (Test)") == "page-a-b-test"

    def test_leading_trailing_dashes(self) -> None:
        assert _sanitize_filename("--Hello World--") == "hello-world"

    def test_consecutive_specials(self) -> None:
        assert _sanitize_filename("A!!!B") == "a-b"

    def test_unicode(self) -> None:
        result = _sanitize_filename("Caf\u00e9 & Bistro")
        assert "caf" in result
        assert result == "caf\u00e9-bistro"

    def test_underscores(self) -> None:
        assert _sanitize_filename("hello_world_page") == "hello-world-page"


# ---------------------------------------------------------------------------
# export_page
# ---------------------------------------------------------------------------


class TestExportPage:
    def test_creates_file(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        output_dir: Path,
    ) -> None:
        page = wiki.create_page(
            kb_id=export_kb.id,
            title="Test Page",
            content="# Hello\nSome content.",
            page_type=PageType.entity,
            tags=["tag1", "tag2"],
        )
        path = exporter.export_page(page, output_dir)
        assert path.exists()
        assert path.name == "test-page.md"

    def test_frontmatter_fields(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        output_dir: Path,
    ) -> None:
        page = wiki.create_page(
            kb_id=export_kb.id,
            title="Frontmatter Test",
            content="Body text.",
            page_type=PageType.concept,
            tags=["alpha", "beta"],
        )
        path = exporter.export_page(page, output_dir)
        text = path.read_text(encoding="utf-8")

        assert text.startswith("---\n")
        assert f"id: {page.id}" in text
        assert "title: Frontmatter Test" in text
        assert "page_type: concept" in text
        assert "  - alpha" in text
        assert "  - beta" in text
        assert "created_at:" in text
        assert "updated_at:" in text
        assert "---" in text

    def test_content_preserved(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        output_dir: Path,
    ) -> None:
        page = wiki.create_page(
            kb_id=export_kb.id,
            title="Content Test",
            content="See also [[Other Page]] and [[Another]].",
            page_type=PageType.entity,
        )
        path = exporter.export_page(page, output_dir)
        text = path.read_text(encoding="utf-8")
        assert "[[Other Page]]" in text
        assert "[[Another]]" in text

    def test_frontmatter_with_sources(
        self, test_db: Database, wiki: WikiManager, exporter: MarkdownExporter,
        export_kb: Knowledgebase, output_dir: Path,
    ) -> None:
        src1 = Source(kb_id=export_kb.id, source_type=SourceType.file, uri="/a.txt")
        src2 = Source(kb_id=export_kb.id, source_type=SourceType.file, uri="/b.txt")
        test_db.insert_source(src1)
        test_db.insert_source(src2)

        page = wiki.create_page(
            kb_id=export_kb.id,
            title="With Sources",
            content="body",
            page_type=PageType.entity,
            source_ids=[src1.id, src2.id],
        )
        path = exporter.export_page(page, output_dir)
        text = path.read_text(encoding="utf-8")
        assert f"  - {src1.id}" in text
        assert f"  - {src2.id}" in text

    def test_empty_tags_and_sources(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        output_dir: Path,
    ) -> None:
        page = wiki.create_page(
            kb_id=export_kb.id,
            title="No Tags",
            content="body",
            page_type=PageType.entity,
        )
        path = exporter.export_page(page, output_dir)
        text = path.read_text(encoding="utf-8")
        # Should have empty list indicators.
        assert "tags:" in text
        assert "sources:" in text


# ---------------------------------------------------------------------------
# export (full)
# ---------------------------------------------------------------------------


class TestExport:
    def test_exports_all_pages(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        output_dir: Path,
    ) -> None:
        wiki.create_page(
            kb_id=export_kb.id, title="Page One", content="p1", page_type=PageType.entity
        )
        wiki.create_page(
            kb_id=export_kb.id, title="Page Two", content="p2", page_type=PageType.summary
        )
        paths = exporter.export(export_kb.id, output_dir)
        assert len(paths) == 2
        filenames = {p.name for p in paths}
        assert "page-one.md" in filenames
        assert "page-two.md" in filenames
        assert all(p.exists() for p in paths)

    def test_creates_output_dir(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        tmp_path: Path,
    ) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        wiki.create_page(
            kb_id=export_kb.id, title="Auto Dir", content="body", page_type=PageType.entity
        )
        paths = exporter.export(export_kb.id, nested)
        assert len(paths) == 1
        assert nested.exists()

    def test_export_empty_kb(
        self, exporter: MarkdownExporter, export_kb: Knowledgebase, output_dir: Path
    ) -> None:
        paths = exporter.export(export_kb.id, output_dir)
        assert paths == []


# ---------------------------------------------------------------------------
# export_incremental
# ---------------------------------------------------------------------------


class TestExportIncremental:
    def test_exports_only_modified(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        output_dir: Path,
    ) -> None:
        wiki.create_page(
            kb_id=export_kb.id, title="Old Page", content="old", page_type=PageType.entity
        )

        # Sleep to ensure timestamp difference.
        time.sleep(0.05)
        cutoff = datetime.now(UTC)
        time.sleep(0.05)

        wiki.create_page(
            kb_id=export_kb.id, title="New Page", content="new", page_type=PageType.entity
        )

        paths = exporter.export_incremental(export_kb.id, output_dir, since=cutoff)
        filenames = {p.name for p in paths}
        assert "new-page.md" in filenames
        assert "old-page.md" not in filenames

    def test_exports_all_when_no_since(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        output_dir: Path,
    ) -> None:
        wiki.create_page(
            kb_id=export_kb.id, title="A", content="a", page_type=PageType.entity
        )
        wiki.create_page(
            kb_id=export_kb.id, title="B", content="b", page_type=PageType.entity
        )
        paths = exporter.export_incremental(export_kb.id, output_dir, since=None)
        assert len(paths) == 2

    def test_exports_updated_pages(
        self, wiki: WikiManager, exporter: MarkdownExporter, export_kb: Knowledgebase,
        output_dir: Path,
    ) -> None:
        page = wiki.create_page(
            kb_id=export_kb.id, title="Evolving", content="old", page_type=PageType.entity
        )

        time.sleep(0.05)
        cutoff = datetime.now(UTC)
        time.sleep(0.05)

        wiki.update_page(page.id, content="new content")

        paths = exporter.export_incremental(export_kb.id, output_dir, since=cutoff)
        filenames = {p.name for p in paths}
        assert "evolving.md" in filenames
