"""Markdown export service for Obsidian-compatible wiki pages."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from agent_knowledgebase.models import WikiPage
from agent_knowledgebase.services.wiki import WikiManager

# Characters that are unsafe/undesirable in filenames.
_UNSAFE_CHARS_RE = re.compile(r"[^\w\s-]")
_WHITESPACE_RE = re.compile(r"[\s_]+")


def _sanitize_filename(title: str) -> str:
    """Convert a page title into a filesystem-safe filename (without extension).

    Rules:
    - Replace special characters with ``-``
    - Lowercase
    - Collapse multiple dashes/spaces
    - Strip leading/trailing dashes
    """
    name = title.lower()
    name = _UNSAFE_CHARS_RE.sub("-", name)
    name = _WHITESPACE_RE.sub("-", name)
    # Collapse consecutive dashes.
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-")
    return name


def _build_frontmatter(page: WikiPage) -> str:
    """Build YAML frontmatter for a wiki page."""
    lines: list[str] = ["---"]
    lines.append(f"id: {page.id}")
    lines.append(f"title: {page.title}")
    lines.append(f"page_type: {page.page_type.value}")

    lines.append("tags:")
    if page.tags:
        for tag in page.tags:
            lines.append(f"  - {tag}")
    else:
        lines.append("  []")

    lines.append(f"created_at: {page.created_at.isoformat()}")
    lines.append(f"updated_at: {page.updated_at.isoformat()}")

    lines.append("sources:")
    if page.source_ids:
        for sid in page.source_ids:
            lines.append(f"  - {sid}")
    else:
        lines.append("  []")

    lines.append("---")
    return "\n".join(lines)


class MarkdownExporter:
    """Export wiki pages to Obsidian-compatible markdown files."""

    def __init__(self, wiki: WikiManager) -> None:
        self._wiki = wiki

    def export(self, kb_id: str, output_dir: Path) -> list[Path]:
        """Export all wiki pages to markdown files.

        Returns list of exported file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pages = self._wiki.list_pages(kb_id)
        return [self.export_page(page, output_dir) for page in pages]

    def export_page(self, page: WikiPage, output_dir: Path) -> Path:
        """Export a single page to markdown.

        - Filename: sanitized title + ``.md``
        - YAML frontmatter: id, title, page_type, tags, created_at, updated_at, sources
        - Content: page content with existing wikilinks preserved
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = _sanitize_filename(page.title) + ".md"
        file_path = output_dir / filename

        frontmatter = _build_frontmatter(page)
        full_content = f"{frontmatter}\n{page.content}\n"
        file_path.write_text(full_content, encoding="utf-8")
        return file_path

    def export_incremental(
        self, kb_id: str, output_dir: Path, since: datetime | None = None
    ) -> list[Path]:
        """Export only pages modified since *since* datetime.

        If *since* is ``None``, export all pages.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pages = self._wiki.list_pages(kb_id)

        if since is not None:
            pages = [p for p in pages if p.updated_at >= since]

        return [self.export_page(page, output_dir) for page in pages]
