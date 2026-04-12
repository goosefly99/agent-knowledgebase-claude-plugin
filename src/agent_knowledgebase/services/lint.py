"""Wiki health checks (linter) for agent-knowledgebase."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from agent_knowledgebase.database import Database
from agent_knowledgebase.models import PageType
from agent_knowledgebase.services.wiki import WikiManager

# Regex for ``[[Title]]`` wikilinks.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")


@dataclass
class LintIssue:
    type: str  # "orphan_page", "missing_page", "stale_ref", "coverage_gap", "index_drift"
    severity: str  # "warning", "error"
    page_id: str | None  # affected page, if any
    message: str
    auto_fixable: bool


@dataclass
class LintReport:
    issues: list[LintIssue]
    checked_at: datetime
    page_count: int
    link_count: int


class WikiLinter:
    """Run health checks against the wiki and report issues."""

    def __init__(self, wiki: WikiManager, db: Database) -> None:
        self._wiki = wiki
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lint(self, kb_id: str) -> LintReport:
        """Run all lint checks and return a report."""
        issues: list[LintIssue] = []
        issues.extend(self.check_orphan_pages(kb_id))
        issues.extend(self.check_missing_pages(kb_id))
        issues.extend(self.check_stale_refs(kb_id))
        issues.extend(self.check_coverage_gaps(kb_id))
        issues.extend(self.check_index_drift(kb_id))

        pages = self._wiki.list_pages(kb_id)
        link_count = sum(len(p.outbound_links) for p in pages)

        return LintReport(
            issues=issues,
            checked_at=datetime.now(UTC),
            page_count=len(pages),
            link_count=link_count,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_orphan_pages(self, kb_id: str) -> list[LintIssue]:
        """Find pages with no inbound links (excluding index pages)."""
        orphans = self._wiki.get_orphan_pages(kb_id)
        return [
            LintIssue(
                type="orphan_page",
                severity="warning",
                page_id=page.id,
                message=f"Page '{page.title}' has no inbound links.",
                auto_fixable=False,
            )
            for page in orphans
        ]

    def check_missing_pages(self, kb_id: str) -> list[LintIssue]:
        """Find ``[[wikilinks]]`` in page content that point to non-existent pages."""
        pages = self._wiki.list_pages(kb_id)
        existing_titles = {p.title for p in pages}
        issues: list[LintIssue] = []
        seen_missing: set[str] = set()

        for page in pages:
            for match in _WIKILINK_RE.finditer(page.content):
                title = match.group(1)
                if title not in existing_titles and title not in seen_missing:
                    seen_missing.add(title)
                    issues.append(
                        LintIssue(
                            type="missing_page",
                            severity="warning",
                            page_id=page.id,
                            message=f"Wikilink [[{title}]] points to a non-existent page.",
                            auto_fixable=True,
                        )
                    )
        return issues

    def check_stale_refs(self, kb_id: str) -> list[LintIssue]:
        """Find pages referencing deleted sources (source_ids pointing to missing sources)."""
        pages = self._wiki.list_pages(kb_id)
        issues: list[LintIssue] = []

        for page in pages:
            for source_id in page.source_ids:
                if self._db.get_source(source_id) is None:
                    issues.append(
                        LintIssue(
                            type="stale_ref",
                            severity="warning",
                            page_id=page.id,
                            message=(
                                f"Page '{page.title}' references deleted source '{source_id}'."
                            ),
                            auto_fixable=True,
                        )
                    )
        return issues

    def check_coverage_gaps(self, kb_id: str) -> list[LintIssue]:
        """Find sources that have no associated wiki pages."""
        sources = self._db.list_sources(kb_id)
        pages = self._wiki.list_pages(kb_id)

        # Collect all source IDs that are referenced by at least one page.
        covered_source_ids: set[str] = set()
        for page in pages:
            covered_source_ids.update(page.source_ids)

        issues: list[LintIssue] = []
        for source in sources:
            if source.id not in covered_source_ids:
                issues.append(
                    LintIssue(
                        type="coverage_gap",
                        severity="warning",
                        page_id=None,
                        message=f"Source '{source.uri}' ({source.id}) has no wiki pages.",
                        auto_fixable=False,
                    )
                )
        return issues

    def check_index_drift(self, kb_id: str) -> list[LintIssue]:
        """Check if the index page is out of sync with actual pages."""
        existing_index = self._db.get_wiki_page_by_title(kb_id, "Index")
        if existing_index is None or existing_index.page_type is not PageType.index:
            # No index page exists; not necessarily an issue, but flag it.
            pages = self._wiki.list_pages(kb_id)
            non_index_pages = [p for p in pages if p.page_type is not PageType.index]
            if non_index_pages:
                return [
                    LintIssue(
                        type="index_drift",
                        severity="warning",
                        page_id=None,
                        message="No index page exists, but the KB has pages.",
                        auto_fixable=True,
                    )
                ]
            return []

        # Generate expected index content and compare.
        expected_index = self._wiki.generate_index(kb_id)
        # After generate_index the index is now up-to-date, so re-fetch.
        refreshed = self._db.get_wiki_page_by_title(kb_id, "Index")
        if refreshed is None:
            return []  # pragma: no cover

        # Compare what the index had before with the newly-generated content.
        if existing_index.content != expected_index.content:
            return [
                LintIssue(
                    type="index_drift",
                    severity="warning",
                    page_id=existing_index.id,
                    message="Index page is out of sync with actual pages.",
                    auto_fixable=True,
                )
            ]
        return []

    # ------------------------------------------------------------------
    # Auto-fix
    # ------------------------------------------------------------------

    def auto_fix(self, kb_id: str, issues: list[LintIssue] | None = None) -> list[LintIssue]:
        """Auto-fix fixable issues. Returns list of issues that were fixed."""
        if issues is None:
            report = self.lint(kb_id)
            issues = report.issues

        fixed: list[LintIssue] = []
        for issue in issues:
            if not issue.auto_fixable:
                continue

            if issue.type == "missing_page":
                self._fix_missing_page(kb_id, issue)
                fixed.append(issue)

            elif issue.type == "stale_ref":
                self._fix_stale_ref(issue)
                fixed.append(issue)

            elif issue.type == "index_drift":
                self._wiki.generate_index(kb_id)
                fixed.append(issue)

        return fixed

    # ------------------------------------------------------------------
    # Fix helpers
    # ------------------------------------------------------------------

    def _fix_missing_page(self, kb_id: str, issue: LintIssue) -> None:
        """Create a stub page for a missing wikilink target."""
        # Extract the title from the issue message.
        match = re.search(r"\[\[(.+?)\]\]", issue.message)
        if match is None:
            return  # pragma: no cover
        title = match.group(1)

        # Only create if it still doesn't exist.
        if self._wiki.get_page_by_title(kb_id, title) is not None:
            return
        self._wiki.create_page(
            kb_id=kb_id,
            title=title,
            content=f"# {title}\n\n_Stub page — auto-generated by linter._\n",
            page_type=PageType.entity,
        )

    def _fix_stale_ref(self, issue: LintIssue) -> None:
        """Remove the stale source association from the page."""
        if issue.page_id is None:
            return  # pragma: no cover

        # Extract source_id from the issue message.
        match = re.search(r"source '([^']+)'", issue.message)
        if match is None:
            return  # pragma: no cover
        source_id = match.group(1)

        self._db.remove_page_source(issue.page_id, source_id)
