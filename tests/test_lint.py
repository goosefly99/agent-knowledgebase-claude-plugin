"""Tests for agent_knowledgebase.services.lint (WikiLinter)."""

from __future__ import annotations

import pytest

from agent_knowledgebase.database import Database
from agent_knowledgebase.models import Knowledgebase, PageType, Source, SourceType
from agent_knowledgebase.services.lint import LintIssue, LintReport, WikiLinter
from agent_knowledgebase.services.wiki import WikiManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def lint_kb(test_db: Database) -> Knowledgebase:
    """Insert and return a knowledgebase for lint tests."""
    kb = Knowledgebase(name="lint-test-kb")
    test_db.insert_knowledgebase(kb)
    return kb


@pytest.fixture()
def wiki(test_db: Database) -> WikiManager:
    return WikiManager(test_db)


@pytest.fixture()
def linter(wiki: WikiManager, test_db: Database) -> WikiLinter:
    return WikiLinter(wiki, test_db)


# ---------------------------------------------------------------------------
# check_orphan_pages
# ---------------------------------------------------------------------------


class TestCheckOrphanPages:
    def test_detects_orphans(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        p1 = wiki.create_page(
            kb_id=lint_kb.id, title="Linked", content="l", page_type=PageType.entity
        )
        wiki.create_page(
            kb_id=lint_kb.id, title="Orphan", content="o", page_type=PageType.entity
        )
        linker = wiki.create_page(
            kb_id=lint_kb.id, title="Linker", content="k", page_type=PageType.entity
        )
        wiki.add_link(linker.id, p1.id)

        issues = linter.check_orphan_pages(lint_kb.id)
        titles = [i.message for i in issues]
        assert any("Orphan" in t for t in titles)
        assert any("Linker" in t for t in titles)
        assert all(i.type == "orphan_page" for i in issues)
        assert all(i.severity == "warning" for i in issues)
        assert all(not i.auto_fixable for i in issues)

    def test_no_orphans(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        p1 = wiki.create_page(
            kb_id=lint_kb.id, title="A", content="a", page_type=PageType.entity
        )
        p2 = wiki.create_page(
            kb_id=lint_kb.id, title="B", content="b", page_type=PageType.entity
        )
        wiki.add_link(p1.id, p2.id)
        wiki.add_link(p2.id, p1.id)

        issues = linter.check_orphan_pages(lint_kb.id)
        assert issues == []

    def test_index_pages_excluded(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=lint_kb.id, title="Index", content="idx", page_type=PageType.index
        )
        issues = linter.check_orphan_pages(lint_kb.id)
        assert all(i.page_id is not None for i in issues)


# ---------------------------------------------------------------------------
# check_missing_pages
# ---------------------------------------------------------------------------


class TestCheckMissingPages:
    def test_detects_missing_links(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Page A",
            content="See also [[Page B]] and [[Nonexistent Page]].",
            page_type=PageType.entity,
        )
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Page B",
            content="body",
            page_type=PageType.entity,
        )

        issues = linter.check_missing_pages(lint_kb.id)
        assert len(issues) == 1
        assert issues[0].type == "missing_page"
        assert "Nonexistent Page" in issues[0].message
        assert issues[0].auto_fixable is True

    def test_no_missing_links(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Alpha",
            content="See [[Beta]].",
            page_type=PageType.entity,
        )
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Beta",
            content="See [[Alpha]].",
            page_type=PageType.entity,
        )
        issues = linter.check_missing_pages(lint_kb.id)
        assert issues == []

    def test_deduplicates_missing(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        """Multiple wikilinks to the same missing page produce one issue."""
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Page A",
            content="[[Missing]] and also [[Missing]].",
            page_type=PageType.entity,
        )
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Page B",
            content="Also links to [[Missing]].",
            page_type=PageType.entity,
        )
        issues = linter.check_missing_pages(lint_kb.id)
        assert len(issues) == 1


# ---------------------------------------------------------------------------
# check_stale_refs
# ---------------------------------------------------------------------------


class TestCheckStaleRefs:
    def test_detects_stale_source(
        self, test_db: Database, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        src = Source(kb_id=lint_kb.id, source_type=SourceType.file, uri="/a.txt")
        test_db.insert_source(src)
        page = wiki.create_page(
            kb_id=lint_kb.id,
            title="Sourced",
            content="body",
            page_type=PageType.entity,
            source_ids=[src.id],
        )
        # Simulate a stale reference: disable FK enforcement, delete the
        # source row, then re-enable FK enforcement.  The page_source row
        # now points to a non-existent source.
        test_db._conn.execute("PRAGMA foreign_keys=OFF")
        test_db.delete_source(src.id)
        test_db._conn.execute("PRAGMA foreign_keys=ON")

        issues = linter.check_stale_refs(lint_kb.id)
        assert len(issues) == 1
        assert issues[0].type == "stale_ref"
        assert issues[0].page_id == page.id
        assert issues[0].auto_fixable is True

    def test_no_stale_refs(
        self, test_db: Database, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        src = Source(kb_id=lint_kb.id, source_type=SourceType.file, uri="/b.txt")
        test_db.insert_source(src)
        wiki.create_page(
            kb_id=lint_kb.id,
            title="OK Page",
            content="body",
            page_type=PageType.entity,
            source_ids=[src.id],
        )
        issues = linter.check_stale_refs(lint_kb.id)
        assert issues == []


# ---------------------------------------------------------------------------
# check_coverage_gaps
# ---------------------------------------------------------------------------


class TestCheckCoverageGaps:
    def test_detects_uncovered_source(
        self, test_db: Database, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        src = Source(kb_id=lint_kb.id, source_type=SourceType.file, uri="/orphan_src.txt")
        test_db.insert_source(src)
        # No pages reference this source.
        issues = linter.check_coverage_gaps(lint_kb.id)
        assert len(issues) == 1
        assert issues[0].type == "coverage_gap"
        assert src.id in issues[0].message
        assert not issues[0].auto_fixable

    def test_covered_source(
        self, test_db: Database, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        src = Source(kb_id=lint_kb.id, source_type=SourceType.file, uri="/covered.txt")
        test_db.insert_source(src)
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Covered Page",
            content="body",
            page_type=PageType.entity,
            source_ids=[src.id],
        )
        issues = linter.check_coverage_gaps(lint_kb.id)
        assert issues == []


# ---------------------------------------------------------------------------
# check_index_drift
# ---------------------------------------------------------------------------


class TestCheckIndexDrift:
    def test_detects_drift(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        """Create an index, then add a page. Index should be stale."""
        wiki.create_page(
            kb_id=lint_kb.id, title="First", content="f", page_type=PageType.entity
        )
        wiki.generate_index(lint_kb.id)

        # Add another page without regenerating the index.
        wiki.create_page(
            kb_id=lint_kb.id, title="Second", content="s", page_type=PageType.entity
        )

        # Manually re-read the index so we see stale content.
        # check_index_drift calls generate_index internally which updates,
        # but compares old content to new. Since there's a new page, old != new.
        # We need to store the old content before the check updates it.
        # Actually, the check itself regenerates. Let's verify the flow:
        # The check gets the current index content, then calls generate_index,
        # which updates it to include "Second". Then it compares old vs new.
        issues = linter.check_index_drift(lint_kb.id)
        assert len(issues) == 1
        assert issues[0].type == "index_drift"
        assert issues[0].auto_fixable is True

    def test_no_drift(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=lint_kb.id, title="Only Page", content="o", page_type=PageType.entity
        )
        wiki.generate_index(lint_kb.id)
        # Index matches actual state.
        issues = linter.check_index_drift(lint_kb.id)
        assert issues == []

    def test_no_index_with_pages(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        """If there are pages but no index, flag drift."""
        wiki.create_page(
            kb_id=lint_kb.id, title="Lonely", content="l", page_type=PageType.entity
        )
        issues = linter.check_index_drift(lint_kb.id)
        assert len(issues) == 1
        assert issues[0].type == "index_drift"

    def test_no_index_no_pages(
        self, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        """Empty KB with no index should not flag drift."""
        issues = linter.check_index_drift(lint_kb.id)
        assert issues == []


# ---------------------------------------------------------------------------
# lint (all checks combined)
# ---------------------------------------------------------------------------


class TestLint:
    def test_lint_returns_report(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Page A",
            content="Links to [[Missing]].",
            page_type=PageType.entity,
        )
        report = linter.lint(lint_kb.id)
        assert isinstance(report, LintReport)
        assert report.page_count >= 1
        assert report.checked_at is not None
        # Should have at least orphan + missing_page issues.
        issue_types = {i.type for i in report.issues}
        assert "orphan_page" in issue_types
        assert "missing_page" in issue_types

    def test_lint_empty_kb(self, linter: WikiLinter, lint_kb: Knowledgebase) -> None:
        report = linter.lint(lint_kb.id)
        assert report.issues == []
        assert report.page_count == 0
        assert report.link_count == 0


# ---------------------------------------------------------------------------
# auto_fix
# ---------------------------------------------------------------------------


class TestAutoFix:
    def test_fix_missing_page(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Referrer",
            content="See [[Brand New Page]].",
            page_type=PageType.entity,
        )
        issues = linter.check_missing_pages(lint_kb.id)
        assert len(issues) == 1

        fixed = linter.auto_fix(lint_kb.id, issues)
        assert len(fixed) == 1

        # The stub page should now exist.
        stub = wiki.get_page_by_title(lint_kb.id, "Brand New Page")
        assert stub is not None
        assert "Stub page" in stub.content

        # Re-check should find no missing pages.
        issues_after = linter.check_missing_pages(lint_kb.id)
        assert issues_after == []

    def test_fix_stale_ref(
        self, test_db: Database, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        src = Source(kb_id=lint_kb.id, source_type=SourceType.file, uri="/del.txt")
        test_db.insert_source(src)
        page = wiki.create_page(
            kb_id=lint_kb.id,
            title="Stale",
            content="body",
            page_type=PageType.entity,
            source_ids=[src.id],
        )
        # Simulate stale ref by deleting source with FK enforcement off.
        test_db._conn.execute("PRAGMA foreign_keys=OFF")
        test_db.delete_source(src.id)
        test_db._conn.execute("PRAGMA foreign_keys=ON")

        issues = linter.check_stale_refs(lint_kb.id)
        assert len(issues) == 1

        fixed = linter.auto_fix(lint_kb.id, issues)
        assert len(fixed) == 1

        # The page_source association should be removed.
        sources = test_db.get_page_sources(page.id)
        assert src.id not in sources

    def test_fix_index_drift(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        wiki.create_page(
            kb_id=lint_kb.id, title="A", content="a", page_type=PageType.entity
        )
        wiki.generate_index(lint_kb.id)
        wiki.create_page(
            kb_id=lint_kb.id, title="B", content="b", page_type=PageType.entity
        )

        issues = linter.check_index_drift(lint_kb.id)
        assert len(issues) == 1

        fixed = linter.auto_fix(lint_kb.id, issues)
        assert len(fixed) == 1

        # Index should now include page B.
        index = wiki.get_page_by_title(lint_kb.id, "Index")
        assert index is not None
        assert "[[B]]" in index.content

    def test_auto_fix_without_explicit_issues(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        """When no issues list is provided, auto_fix runs lint first."""
        wiki.create_page(
            kb_id=lint_kb.id,
            title="Has Link",
            content="Go to [[Phantom Page]].",
            page_type=PageType.entity,
        )
        fixed = linter.auto_fix(lint_kb.id)
        assert any(i.type == "missing_page" for i in fixed)

    def test_non_fixable_issues_skipped(
        self, wiki: WikiManager, linter: WikiLinter, lint_kb: Knowledgebase
    ) -> None:
        """auto_fix should skip non-fixable issues."""
        issues = [
            LintIssue(
                type="orphan_page",
                severity="warning",
                page_id="some-id",
                message="Page 'X' has no inbound links.",
                auto_fixable=False,
            ),
        ]
        fixed = linter.auto_fix(lint_kb.id, issues)
        assert fixed == []
