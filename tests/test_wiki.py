"""Tests for agent_knowledgebase.services.wiki (WikiManager)."""

from __future__ import annotations

import time

import pytest

from agent_knowledgebase.database import Database
from agent_knowledgebase.models import Knowledgebase, PageType, Source, SourceType, WikiPage
from agent_knowledgebase.services.wiki import WikiManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wiki_kb(test_db: Database) -> Knowledgebase:
    """Insert and return a knowledgebase for wiki tests."""
    kb = Knowledgebase(name="wiki-test-kb")
    test_db.insert_knowledgebase(kb)
    return kb


@pytest.fixture()
def wiki(test_db: Database) -> WikiManager:
    """Return a WikiManager backed by the test database."""
    return WikiManager(test_db)


# ---------------------------------------------------------------------------
# Create page
# ---------------------------------------------------------------------------


class TestCreatePage:
    def test_create_and_persist(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="My Entity",
            content="Some content",
            page_type=PageType.entity,
            tags=["tag1", "tag2"],
        )
        assert page.id is not None
        assert page.title == "My Entity"
        assert page.page_type is PageType.entity
        assert page.tags == ["tag1", "tag2"]

        fetched = wiki.get_page(page.id)
        assert fetched is not None
        assert fetched.title == "My Entity"
        assert fetched.content == "Some content"

    def test_create_with_sources(
        self, test_db: Database, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        src = Source(kb_id=wiki_kb.id, source_type=SourceType.file, uri="/tmp/f.txt")
        test_db.insert_source(src)

        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Sourced Page",
            content="body",
            page_type=PageType.summary,
            source_ids=[src.id],
        )
        fetched = wiki.get_page(page.id)
        assert fetched is not None
        assert src.id in fetched.source_ids

    def test_create_defaults(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Defaults",
            content="body",
            page_type=PageType.concept,
        )
        assert page.tags == []
        assert page.source_ids == []


# ---------------------------------------------------------------------------
# Get page by title
# ---------------------------------------------------------------------------


class TestGetPageByTitle:
    def test_found(self, wiki: WikiManager, wiki_kb: Knowledgebase) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id,
            title="Unique Title",
            content="body",
            page_type=PageType.entity,
        )
        found = wiki.get_page_by_title(wiki_kb.id, "Unique Title")
        assert found is not None
        assert found.title == "Unique Title"

    def test_not_found(self, wiki: WikiManager, wiki_kb: Knowledgebase) -> None:
        result = wiki.get_page_by_title(wiki_kb.id, "No Such Page")
        assert result is None

    def test_scoped_to_kb(
        self, test_db: Database, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id,
            title="Shared Name",
            content="body",
            page_type=PageType.entity,
        )
        other_kb = Knowledgebase(name="other-kb")
        test_db.insert_knowledgebase(other_kb)
        assert wiki.get_page_by_title(other_kb.id, "Shared Name") is None


# ---------------------------------------------------------------------------
# Update page
# ---------------------------------------------------------------------------


class TestUpdatePage:
    def test_update_content(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="To Update",
            content="old",
            page_type=PageType.entity,
        )
        original_updated = page.updated_at

        # Ensure the timestamp advances.
        time.sleep(0.05)

        updated = wiki.update_page(page.id, content="new content")
        assert updated.content == "new content"
        assert updated.title == "To Update"  # unchanged
        assert updated.updated_at > original_updated

        fetched = wiki.get_page(page.id)
        assert fetched is not None
        assert fetched.content == "new content"

    def test_update_title(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Old Title",
            content="body",
            page_type=PageType.entity,
        )
        updated = wiki.update_page(page.id, title="New Title")
        assert updated.title == "New Title"

    def test_update_tags(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Tagged",
            content="body",
            page_type=PageType.entity,
            tags=["old"],
        )
        updated = wiki.update_page(page.id, tags=["new1", "new2"])
        assert updated.tags == ["new1", "new2"]

        fetched = wiki.get_page(page.id)
        assert fetched is not None
        assert fetched.tags == ["new1", "new2"]

    def test_update_nonexistent_raises(self, wiki: WikiManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            wiki.update_page("nonexistent-id", content="x")


# ---------------------------------------------------------------------------
# Delete page
# ---------------------------------------------------------------------------


class TestDeletePage:
    def test_delete(self, wiki: WikiManager, wiki_kb: Knowledgebase) -> None:
        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Doomed",
            content="body",
            page_type=PageType.entity,
        )
        wiki.delete_page(page.id)
        assert wiki.get_page(page.id) is None

    def test_delete_removes_links(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        p1 = wiki.create_page(
            kb_id=wiki_kb.id, title="A", content="a", page_type=PageType.entity
        )
        p2 = wiki.create_page(
            kb_id=wiki_kb.id, title="B", content="b", page_type=PageType.entity
        )
        wiki.add_link(p1.id, p2.id)
        wiki.delete_page(p1.id)
        # p2 should no longer see the inbound link.
        linked = wiki.get_linked_pages(p2.id, direction="inbound")
        assert len(linked) == 0


# ---------------------------------------------------------------------------
# List pages
# ---------------------------------------------------------------------------


class TestListPages:
    def test_list_all(self, wiki: WikiManager, wiki_kb: Knowledgebase) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id, title="E1", content="e", page_type=PageType.entity
        )
        wiki.create_page(
            kb_id=wiki_kb.id, title="S1", content="s", page_type=PageType.summary
        )
        pages = wiki.list_pages(wiki_kb.id)
        assert len(pages) == 2

    def test_list_by_type(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id, title="E1", content="e", page_type=PageType.entity
        )
        wiki.create_page(
            kb_id=wiki_kb.id, title="E2", content="e2", page_type=PageType.entity
        )
        wiki.create_page(
            kb_id=wiki_kb.id, title="S1", content="s", page_type=PageType.summary
        )
        entities = wiki.list_pages(wiki_kb.id, page_type=PageType.entity)
        assert len(entities) == 2
        assert all(p.page_type is PageType.entity for p in entities)

    def test_list_empty(
        self, test_db: Database, wiki: WikiManager
    ) -> None:
        other_kb = Knowledgebase(name="empty-kb")
        test_db.insert_knowledgebase(other_kb)
        assert wiki.list_pages(other_kb.id) == []


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_by_content(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id,
            title="Python Guide",
            content="Python is a versatile programming language.",
            page_type=PageType.summary,
        )
        results = wiki.search("versatile programming", wiki_kb.id)
        assert len(results) >= 1
        assert results[0].title == "Python Guide"

    def test_search_by_title(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id,
            title="Concurrency Patterns",
            content="body text",
            page_type=PageType.concept,
        )
        results = wiki.search("Concurrency", wiki_kb.id)
        assert len(results) >= 1

    def test_search_no_results(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id,
            title="Something",
            content="body",
            page_type=PageType.entity,
        )
        results = wiki.search("zzzznonexistent", wiki_kb.id)
        assert results == []

    def test_search_after_update(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        """FTS index should reflect content changes after update."""
        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Evolving",
            content="alpha beta",
            page_type=PageType.entity,
        )
        wiki.update_page(page.id, content="gamma delta")
        # Old content should no longer match.
        assert wiki.search("alpha", wiki_kb.id) == []
        # New content should match.
        results = wiki.search("gamma", wiki_kb.id)
        assert len(results) == 1
        assert results[0].id == page.id


# ---------------------------------------------------------------------------
# Link management
# ---------------------------------------------------------------------------


class TestLinkManagement:
    def test_add_and_get_outbound(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        p1 = wiki.create_page(
            kb_id=wiki_kb.id, title="Source", content="s", page_type=PageType.entity
        )
        p2 = wiki.create_page(
            kb_id=wiki_kb.id, title="Target", content="t", page_type=PageType.entity
        )
        wiki.add_link(p1.id, p2.id)

        outbound = wiki.get_linked_pages(p1.id, direction="outbound")
        assert len(outbound) == 1
        assert outbound[0].id == p2.id
        assert outbound[0].title == "Target"

    def test_add_and_get_inbound(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        p1 = wiki.create_page(
            kb_id=wiki_kb.id, title="Source", content="s", page_type=PageType.entity
        )
        p2 = wiki.create_page(
            kb_id=wiki_kb.id, title="Target", content="t", page_type=PageType.entity
        )
        wiki.add_link(p1.id, p2.id)

        inbound = wiki.get_linked_pages(p2.id, direction="inbound")
        assert len(inbound) == 1
        assert inbound[0].id == p1.id

    def test_remove_link(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        p1 = wiki.create_page(
            kb_id=wiki_kb.id, title="A", content="a", page_type=PageType.entity
        )
        p2 = wiki.create_page(
            kb_id=wiki_kb.id, title="B", content="b", page_type=PageType.entity
        )
        wiki.add_link(p1.id, p2.id)
        wiki.remove_link(p1.id, p2.id)
        assert wiki.get_linked_pages(p1.id, direction="outbound") == []

    def test_linked_pages_returns_full_objects(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        p1 = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Detailed Source",
            content="rich content",
            page_type=PageType.entity,
            tags=["important"],
        )
        p2 = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Detailed Target",
            content="also rich",
            page_type=PageType.concept,
        )
        wiki.add_link(p1.id, p2.id)
        linked = wiki.get_linked_pages(p1.id)
        assert len(linked) == 1
        assert isinstance(linked[0], WikiPage)
        assert linked[0].content == "also rich"
        assert linked[0].page_type is PageType.concept


# ---------------------------------------------------------------------------
# Orphan page detection
# ---------------------------------------------------------------------------


class TestOrphanPages:
    def test_orphan_detection(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        p1 = wiki.create_page(
            kb_id=wiki_kb.id, title="Linked", content="l", page_type=PageType.entity
        )
        p2 = wiki.create_page(
            kb_id=wiki_kb.id, title="Orphan", content="o", page_type=PageType.entity
        )
        p3 = wiki.create_page(
            kb_id=wiki_kb.id, title="Linker", content="k", page_type=PageType.entity
        )
        wiki.add_link(p3.id, p1.id)

        orphans = wiki.get_orphan_pages(wiki_kb.id)
        orphan_ids = {p.id for p in orphans}
        # p2 and p3 have no inbound links; p1 has one.
        assert p2.id in orphan_ids
        assert p3.id in orphan_ids
        assert p1.id not in orphan_ids

    def test_index_pages_excluded(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id,
            title="Index",
            content="index body",
            page_type=PageType.index,
        )
        wiki.create_page(
            kb_id=wiki_kb.id,
            title="Orphan Entity",
            content="body",
            page_type=PageType.entity,
        )
        orphans = wiki.get_orphan_pages(wiki_kb.id)
        assert all(p.page_type is not PageType.index for p in orphans)
        assert len(orphans) == 1
        assert orphans[0].title == "Orphan Entity"


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------


class TestGenerateIndex:
    def test_create_index(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id, title="Alpha", content="a", page_type=PageType.entity
        )
        wiki.create_page(
            kb_id=wiki_kb.id, title="Beta", content="b", page_type=PageType.concept
        )
        index = wiki.generate_index(wiki_kb.id)
        assert index.title == "Index"
        assert index.page_type is PageType.index
        assert "[[Alpha]]" in index.content
        assert "[[Beta]]" in index.content
        assert "## Entity" in index.content
        assert "## Concept" in index.content

    def test_update_existing_index(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id, title="First", content="f", page_type=PageType.entity
        )
        index1 = wiki.generate_index(wiki_kb.id)
        assert "[[First]]" in index1.content

        # Add another page and regenerate.
        wiki.create_page(
            kb_id=wiki_kb.id, title="Second", content="s", page_type=PageType.entity
        )
        index2 = wiki.generate_index(wiki_kb.id)

        # Same page ID (updated, not duplicated).
        assert index2.id == index1.id
        assert "[[First]]" in index2.content
        assert "[[Second]]" in index2.content

    def test_index_excludes_index_pages(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id, title="Entity A", content="a", page_type=PageType.entity
        )
        index = wiki.generate_index(wiki_kb.id)
        # The index page itself should not be listed.
        assert "[[Index]]" not in index.content

    def test_index_organizes_by_type(
        self, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=wiki_kb.id, title="E1", content="e", page_type=PageType.entity
        )
        wiki.create_page(
            kb_id=wiki_kb.id, title="C1", content="c", page_type=PageType.comparison
        )
        wiki.create_page(
            kb_id=wiki_kb.id, title="S1", content="s", page_type=PageType.synthesis
        )
        index = wiki.generate_index(wiki_kb.id)
        assert "## Entity" in index.content
        assert "## Comparison" in index.content
        assert "## Synthesis" in index.content


# ---------------------------------------------------------------------------
# Source association
# ---------------------------------------------------------------------------


class TestSourceAssociation:
    def test_add_and_get_sources(
        self, test_db: Database, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        src1 = Source(kb_id=wiki_kb.id, source_type=SourceType.file, uri="/a.txt")
        src2 = Source(kb_id=wiki_kb.id, source_type=SourceType.file, uri="/b.txt")
        test_db.insert_source(src1)
        test_db.insert_source(src2)

        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Sourced",
            content="body",
            page_type=PageType.entity,
        )
        wiki.add_source_to_page(page.id, src1.id)
        wiki.add_source_to_page(page.id, src2.id)

        sources = wiki.get_page_sources(page.id)
        assert src1.id in sources
        assert src2.id in sources

    def test_create_with_sources_then_add_more(
        self, test_db: Database, wiki: WikiManager, wiki_kb: Knowledgebase
    ) -> None:
        src1 = Source(kb_id=wiki_kb.id, source_type=SourceType.file, uri="/x.txt")
        src2 = Source(kb_id=wiki_kb.id, source_type=SourceType.file, uri="/y.txt")
        test_db.insert_source(src1)
        test_db.insert_source(src2)

        page = wiki.create_page(
            kb_id=wiki_kb.id,
            title="Multi Source",
            content="body",
            page_type=PageType.entity,
            source_ids=[src1.id],
        )
        wiki.add_source_to_page(page.id, src2.id)

        sources = wiki.get_page_sources(page.id)
        assert len(sources) == 2
        assert src1.id in sources
        assert src2.id in sources
