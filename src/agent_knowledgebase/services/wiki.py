"""Wiki artifact manager for structured wiki pages."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_knowledgebase.database import Database
from agent_knowledgebase.models import PageType, WikiPage


class WikiManager:
    """Manages wiki page lifecycle, links, and index generation.

    All mutations are persisted to the database immediately.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Page CRUD
    # ------------------------------------------------------------------

    def create_page(
        self,
        kb_id: str,
        title: str,
        content: str,
        page_type: PageType,
        tags: list[str] | None = None,
        source_ids: list[str] | None = None,
    ) -> WikiPage:
        """Create a new wiki page and persist it."""
        page = WikiPage(
            kb_id=kb_id,
            title=title,
            content=content,
            page_type=page_type,
            tags=tags or [],
            source_ids=source_ids or [],
        )
        self._db.insert_wiki_page(page)
        return page

    def get_page(self, page_id: str) -> WikiPage | None:
        """Get a page by ID."""
        return self._db.get_wiki_page(page_id)

    def get_page_by_title(self, kb_id: str, title: str) -> WikiPage | None:
        """Find a page by its title within a KB."""
        return self._db.get_wiki_page_by_title(kb_id, title)

    def update_page(
        self,
        page_id: str,
        content: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> WikiPage:
        """Update a page's content, title, and/or tags.

        Updates the ``updated_at`` timestamp.

        Raises
        ------
        ValueError
            If the page does not exist.
        """
        page = self._db.get_wiki_page(page_id)
        if page is None:
            raise ValueError(f"Wiki page not found: {page_id}")

        if content is not None:
            page.content = content
        if title is not None:
            page.title = title
        if tags is not None:
            page.tags = tags

        page.updated_at = datetime.now(UTC)
        self._db.update_wiki_page(page)
        return page

    def delete_page(self, page_id: str) -> None:
        """Delete a page and its links."""
        self._db.delete_wiki_page(page_id)

    def list_pages(
        self, kb_id: str, page_type: PageType | None = None
    ) -> list[WikiPage]:
        """List pages in a KB, optionally filtered by type."""
        if page_type is not None:
            return self._db.list_wiki_pages_by_type(kb_id, page_type.value)
        return self._db.list_wiki_pages(kb_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, kb_id: str) -> list[WikiPage]:
        """Full-text search across wiki pages."""
        return self._db.search_wiki_fts(query, kb_id)

    # ------------------------------------------------------------------
    # Link management
    # ------------------------------------------------------------------

    def add_link(self, from_page_id: str, to_page_id: str) -> None:
        """Add a link between pages."""
        self._db.add_wiki_link(from_page_id, to_page_id)

    def remove_link(self, from_page_id: str, to_page_id: str) -> None:
        """Remove a link between pages."""
        self._db.remove_wiki_link(from_page_id, to_page_id)

    def get_linked_pages(
        self, page_id: str, direction: str = "outbound"
    ) -> list[WikiPage]:
        """Get pages linked to/from a page.

        Returns full :class:`WikiPage` objects, not just IDs.

        Parameters
        ----------
        page_id:
            The page whose links are queried.
        direction:
            ``"outbound"`` (default) or ``"inbound"``.
        """
        linked_ids = self._db.get_wiki_links(page_id, direction)  # type: ignore[arg-type]
        pages: list[WikiPage] = []
        for lid in linked_ids:
            page = self._db.get_wiki_page(lid)
            if page is not None:
                pages.append(page)
        return pages

    def get_orphan_pages(self, kb_id: str) -> list[WikiPage]:
        """Find pages with no inbound links, excluding index pages."""
        all_pages = self._db.list_wiki_pages(kb_id)
        orphans: list[WikiPage] = []
        for page in all_pages:
            if page.page_type is PageType.index:
                continue
            if not page.inbound_links:
                orphans.append(page)
        return orphans

    # ------------------------------------------------------------------
    # Index generation
    # ------------------------------------------------------------------

    def generate_index(self, kb_id: str) -> WikiPage:
        """Generate or update the index page for a KB.

        Lists all pages organized by :class:`PageType` with ``[[Title]]``
        wikilinks.  If an index page already exists, it is updated;
        otherwise a new one is created.
        """
        all_pages = self._db.list_wiki_pages(kb_id)

        # Group pages by type (excluding index pages from the listing).
        by_type: dict[PageType, list[WikiPage]] = {}
        for page in all_pages:
            if page.page_type is PageType.index:
                continue
            by_type.setdefault(page.page_type, []).append(page)

        # Build markdown content.
        lines: list[str] = ["# Index", ""]
        for page_type in PageType:
            if page_type is PageType.index:
                continue
            pages_of_type = by_type.get(page_type, [])
            if not pages_of_type:
                continue
            lines.append(f"## {page_type.value.title()}")
            lines.append("")
            for page in sorted(pages_of_type, key=lambda p: p.title):
                lines.append(f"- [[{page.title}]]")
            lines.append("")

        content = "\n".join(lines)

        # Check for existing index page.
        existing = self._db.get_wiki_page_by_title(kb_id, "Index")
        if existing is not None and existing.page_type is PageType.index:
            existing.content = content
            existing.updated_at = datetime.now(UTC)
            self._db.update_wiki_page(existing)
            return existing

        # Create new index page.
        index_page = WikiPage(
            kb_id=kb_id,
            title="Index",
            content=content,
            page_type=PageType.index,
        )
        self._db.insert_wiki_page(index_page)
        return index_page

    # ------------------------------------------------------------------
    # Source associations
    # ------------------------------------------------------------------

    def add_source_to_page(self, page_id: str, source_id: str) -> None:
        """Associate a source with a page."""
        self._db.add_page_source(page_id, source_id)

    def get_page_sources(self, page_id: str) -> list[str]:
        """Get source IDs for a page."""
        return self._db.get_page_sources(page_id)
