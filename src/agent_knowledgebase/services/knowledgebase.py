"""Top-level knowledgebase lifecycle service coordinating all sub-services.

Each knowledgebase is stored in its own subdirectory under
``<saves_dir>/<sanitized-name>/``, containing a per-KB SQLite database
and ChromaDB directory.  A lightweight ``.index.json`` at the
``saves_dir`` level maps ``kb_id -> dir_name`` for fast lookups.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from agent_knowledgebase.config import Settings, sanitize_kb_dir_name
from agent_knowledgebase.database import Database
from agent_knowledgebase.models import (
    Knowledgebase,
    PageType,
    PipelineRun,
    Source,
    SourceStatus,
    SourceType,
    WikiPage,
)
from agent_knowledgebase.services.dedup_service import DEFAULT_DEDUP_POLICY, DedupPolicy, resolve_dedup_action
from agent_knowledgebase.services.embeddings import Embedder, create_embedder
from agent_knowledgebase.services.export import MarkdownExporter
from agent_knowledgebase.services.ingestion import IngestionOrchestrator
from agent_knowledgebase.services.lint import LintIssue, LintReport, WikiLinter
from agent_knowledgebase.services.pipeline import PipelineManager
from agent_knowledgebase.services.query import (
    QueryOrchestrator,
    SearchResult,
    default_top_k,
    hybrid_weights_from,
)
from agent_knowledgebase.services.vectorstore import VectorStore, create_vectorstore
from agent_knowledgebase.services.wiki import WikiManager


# ---------------------------------------------------------------------------
# Per-KB context bundle
# ---------------------------------------------------------------------------


@dataclass
class _KBContext:
    """Internal bundle of per-KB database, sub-services, and vectorstore."""

    db: Database
    wiki: WikiManager
    pipeline: PipelineManager
    linter: WikiLinter
    exporter: MarkdownExporter
    vectorstore: Optional[VectorStore] = field(default=None)


# ---------------------------------------------------------------------------
# Index file helpers
# ---------------------------------------------------------------------------

_INDEX_FILENAME = ".index.json"


def _load_index(base_dir: Path) -> dict[str, str]:
    """Load the ``kb_id -> dir_name`` index, returning ``{}`` if absent."""
    path = base_dir / _INDEX_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_index(base_dir: Path, index: dict[str, str]) -> None:
    """Atomically write the index file."""
    path = base_dir / _INDEX_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class KnowledgebaseService:
    """Coordinates all sub-services for full knowledgebase lifecycle.

    Each knowledgebase is stored in its own subdirectory with an
    independent SQLite database and ChromaDB directory.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config.resolve_paths()
        self._base_dir = self._config.knowledgebases_dir

        # Shared stateless services — embedder is loaded lazily the first
        # time it is needed since the sentence-transformers model load takes
        # several seconds and would otherwise dominate first-call latency.
        self._embedder_instance: Embedder | None = None
        self._ingestion = IngestionOrchestrator(self._config)

        # Per-KB state — lazily populated
        self._contexts: dict[str, _KBContext] = {}
        self._index: dict[str, str] = _load_index(self._base_dir)

    @property
    def _embedder(self) -> Embedder:
        """Lazily instantiate the embedder on first use."""
        if self._embedder_instance is None:
            self._embedder_instance = create_embedder(self._config)
        return self._embedder_instance

    # ------------------------------------------------------------------
    # Per-KB context management
    # ------------------------------------------------------------------

    def _open_context(self, kb_id: str, dir_name: str) -> _KBContext:
        """Open (or return cached) the per-KB context for *kb_id*."""
        if kb_id in self._contexts:
            return self._contexts[kb_id]
        db = Database(self._config.kb_db_path(dir_name))
        wiki = WikiManager(db)
        ctx = _KBContext(
            db=db,
            wiki=wiki,
            pipeline=PipelineManager(db),
            linter=WikiLinter(wiki, db),
            exporter=MarkdownExporter(wiki),
        )
        self._contexts[kb_id] = ctx
        return ctx

    def _ctx(self, kb_id: str) -> _KBContext:
        """Get the context for *kb_id*, raising if not in the index."""
        dir_name = self._index.get(kb_id)
        if dir_name is None:
            raise ValueError(f"KB {kb_id} not found")
        return self._open_context(kb_id, dir_name)

    def _get_vectorstore(self, kb_id: str) -> VectorStore:
        """Get or create the vectorstore for *kb_id*."""
        ctx = self._ctx(kb_id)
        if ctx.vectorstore is None:
            dir_name = self._index[kb_id]
            ctx.vectorstore = create_vectorstore(
                self._config,
                collection_name=kb_id,
                chroma_path=self._config.kb_chroma_path(dir_name),
            )
        return ctx.vectorstore

    def _find_context_by_source(self, source_id: str) -> tuple[_KBContext, str]:
        """Scan all KBs for a source, returning ``(context, kb_id)``."""
        for kid, dname in self._index.items():
            ctx = self._open_context(kid, dname)
            if ctx.db.get_source(source_id) is not None:
                return ctx, kid
        raise ValueError(f"Source {source_id} not found")

    def _find_context_by_page(self, page_id: str) -> tuple[_KBContext, str]:
        """Scan all KBs for a wiki page, returning ``(context, kb_id)``."""
        for kid, dname in self._index.items():
            ctx = self._open_context(kid, dname)
            if ctx.db.get_wiki_page(page_id) is not None:
                return ctx, kid
        raise ValueError(f"Page {page_id} not found")

    # ------------------------------------------------------------------
    # KB Lifecycle
    # ------------------------------------------------------------------

    def create_kb(self, name: str, description: str = "") -> Knowledgebase:
        """Create a new knowledgebase and its on-disk directory."""
        dir_name = sanitize_kb_dir_name(name)

        # Prevent directory-name collisions
        if dir_name in self._index.values():
            raise ValueError(
                f"A knowledgebase directory '{dir_name}' already exists (from name '{name}')"
            )

        kb = Knowledgebase(name=name, description=description)

        # Create the KB directory and open its database
        self._config.kb_data_dir(dir_name).mkdir(parents=True, exist_ok=True)
        ctx = self._open_context(kb.id, dir_name)
        ctx.db.insert_knowledgebase(kb)

        # Update the index
        self._index[kb.id] = dir_name
        _save_index(self._base_dir, self._index)

        return self._enrich_kb(kb, ctx)

    def get_kb(self, kb_id: str) -> Knowledgebase | None:
        """Get KB by ID, enriched with source_count and page_count."""
        dir_name = self._index.get(kb_id)
        if dir_name is None:
            return None
        ctx = self._open_context(kb_id, dir_name)
        kb = ctx.db.get_knowledgebase(kb_id)
        if kb is None:
            return None
        return self._enrich_kb(kb, ctx)

    def list_kbs(self) -> list[Knowledgebase]:
        """List all KBs, enriched with counts."""
        results: list[Knowledgebase] = []
        for kb_id, dir_name in self._index.items():
            ctx = self._open_context(kb_id, dir_name)
            kb = ctx.db.get_knowledgebase(kb_id)
            if kb is not None:
                results.append(self._enrich_kb(kb, ctx))
        return results

    def delete_kb(self, kb_id: str) -> None:
        """Delete a KB and ALL associated data including its directory."""
        dir_name = self._index.get(kb_id)
        if dir_name is None:
            return  # idempotent

        # Close cached context
        ctx = self._contexts.pop(kb_id, None)
        if ctx is not None:
            ctx.db.close()

        # Remove the entire KB directory from disk
        kb_dir = self._config.kb_data_dir(dir_name)
        if kb_dir.is_dir():
            shutil.rmtree(kb_dir)

        # Update the index
        del self._index[kb_id]
        _save_index(self._base_dir, self._index)

    # ------------------------------------------------------------------
    # Source Ingestion (Pipeline-Gated)
    # ------------------------------------------------------------------

    def ingest_source(
        self,
        kb_id: str,
        source_type: SourceType,
        uri: str,
        metadata: dict | None = None,
        dedup_key: str | None = None,
        dedup_policy: DedupPolicy = DEFAULT_DEDUP_POLICY,
    ) -> Source:
        """Full ingestion pipeline for a new source."""
        ctx = self._ctx(kb_id)

        # Verify KB exists
        kb = ctx.db.get_knowledgebase(kb_id)
        if kb is None:
            raise ValueError(f"KB {kb_id} not found")

        # Dedup check
        action, existing = resolve_dedup_action(ctx.db, kb_id, dedup_key, dedup_policy)
        if action == "skip":
            return existing  # type: ignore[return-value]
        if action == "replace":
            self.remove_source(existing.id)  # type: ignore[union-attr]

        # Create source record
        source = Source(
            kb_id=kb_id,
            source_type=source_type,
            uri=uri,
            metadata=metadata or {},
            status=SourceStatus.ingesting,
            dedup_key=dedup_key or None,
        )
        ctx.db.insert_source(source)

        # Start pipeline
        run = ctx.pipeline.start_run(kb_id, source.id)

        try:
            # Phase 1: initialize
            ctx.pipeline.complete_phase(run.id)
            ctx.pipeline.advance_phase(run.id)

            # Phase 2: read_source
            chunks = self._ingestion.ingest(source_type, uri, metadata)
            for chunk in chunks:
                chunk.source_id = source.id
                chunk.kb_id = kb_id
            ctx.pipeline.complete_phase(run.id)
            ctx.pipeline.advance_phase(run.id)

            # Phase 3: chunk
            ctx.pipeline.complete_phase(run.id)
            ctx.pipeline.advance_phase(run.id)

            # Phase 4: embed
            texts = [c.content for c in chunks]
            embeddings = self._embedder.embed(texts) if texts else []
            vs = self._get_vectorstore(kb_id)
            if chunks:
                vs.add(
                    ids=[c.id for c in chunks],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[c.metadata for c in chunks],
                )
            for chunk in chunks:
                ctx.db.insert_chunk(chunk)
            ctx.pipeline.complete_phase(run.id)
            ctx.pipeline.advance_phase(run.id)

            # Phase 5: integrate_wiki
            summary_content = f"# {uri}\n\nSource type: {source_type.value}\n\n"
            summary_content += f"Chunks ingested: {len(chunks)}\n"
            if chunks:
                summary_content += f"\n## Content Preview\n\n{chunks[0].content[:500]}..."
            ctx.wiki.create_page(
                kb_id=kb_id,
                title=f"Source: {uri}",
                content=summary_content,
                page_type=PageType.summary,
                source_ids=[source.id],
            )
            ctx.pipeline.complete_phase(run.id)
            ctx.pipeline.advance_phase(run.id)

            # Phase 6: finalize
            source.status = SourceStatus.ingested
            source.chunk_count = len(chunks)
            source.ingested_at = datetime.now(UTC)
            ctx.db.update_source(source)
            ctx.pipeline.complete_phase(run.id)

            return source

        except Exception as e:
            ctx.pipeline.fail_phase(run.id, str(e))
            source.status = SourceStatus.failed
            ctx.db.update_source(source)
            raise

    def update_source(self, source_id: str) -> Source:
        """Re-ingest a source (delete old chunks/vectors, re-run pipeline)."""
        ctx, _ = self._find_context_by_source(source_id)
        source = ctx.db.get_source(source_id)
        if source is None:
            raise ValueError(f"Source {source_id} not found")

        # Delete old chunks from vectorstore
        old_chunks = ctx.db.list_chunks(source_id)
        if old_chunks:
            vs = self._get_vectorstore(source.kb_id)
            vs.delete([c.id for c in old_chunks])
        ctx.db.delete_chunks_by_source(source_id)

        return self.ingest_source(source.kb_id, source.source_type, source.uri, source.metadata)

    def remove_source(self, source_id: str) -> None:
        """Remove a source and its chunks/vectors from the KB."""
        ctx, _ = self._find_context_by_source(source_id)
        source = ctx.db.get_source(source_id)
        if source is None:
            raise ValueError(f"Source {source_id} not found")

        old_chunks = ctx.db.list_chunks(source_id)
        if old_chunks:
            vs = self._get_vectorstore(source.kb_id)
            vs.delete([c.id for c in old_chunks])

        ctx.db.delete_pipeline_runs_by_source(source_id)
        ctx.db.delete_chunks_by_source(source_id)
        ctx.db.delete_page_sources_by_source(source_id)
        ctx.db.delete_source(source_id)

    # ------------------------------------------------------------------
    # Query (No Pipeline Required)
    # ------------------------------------------------------------------

    def query(self, kb_id: str, text: str, top_k: int | None = None) -> list[SearchResult]:
        """Semantic query across a KB."""
        if top_k is None:
            top_k = default_top_k(self._config)
        ctx = self._ctx(kb_id)
        vs = self._get_vectorstore(kb_id)
        orchestrator = QueryOrchestrator(vs, self._embedder, ctx.wiki)
        return orchestrator.query(text, kb_id, top_k=top_k)

    def search(self, kb_id: str, text: str, top_k: int | None = None) -> list[SearchResult]:
        """Keyword search across a KB."""
        if top_k is None:
            top_k = default_top_k(self._config)
        ctx = self._ctx(kb_id)
        vs = self._get_vectorstore(kb_id)
        orchestrator = QueryOrchestrator(vs, self._embedder, ctx.wiki)
        return orchestrator.search(text, kb_id, top_k=top_k)

    def hybrid_query(self, kb_id: str, text: str, top_k: int | None = None) -> list[SearchResult]:
        """Combined semantic + keyword search."""
        if top_k is None:
            top_k = default_top_k(self._config)
        vector_weight, fts_weight, fetch_mult = hybrid_weights_from(self._config)
        ctx = self._ctx(kb_id)
        vs = self._get_vectorstore(kb_id)
        orchestrator = QueryOrchestrator(vs, self._embedder, ctx.wiki)
        return orchestrator.hybrid_query(
            text,
            kb_id,
            top_k=top_k,
            vector_weight=vector_weight,
            fts_weight=fts_weight,
            fetch_multiplier=fetch_mult,
        )

    # ------------------------------------------------------------------
    # Wiki Operations
    # ------------------------------------------------------------------

    def get_page(self, page_id: str) -> WikiPage | None:
        """Get a wiki page by ID (scans all KBs)."""
        for kb_id, dir_name in self._index.items():
            ctx = self._open_context(kb_id, dir_name)
            page = ctx.wiki.get_page(page_id)
            if page is not None:
                return page
        return None

    def list_pages(self, kb_id: str, page_type: PageType | None = None) -> list[WikiPage]:
        """List wiki pages in a KB, optionally filtered by type."""
        ctx = self._ctx(kb_id)
        pages = ctx.wiki.list_pages(kb_id, page_type)
        # Build a source lookup in one query, then stamp first-source fields onto each page.
        sources = {s.id: s for s in ctx.db.list_sources(kb_id)}
        for page in pages:
            first_source = next((sources[sid] for sid in page.source_ids if sid in sources), None)
            page.source_type = first_source.source_type.value if first_source else None
            page.uri = first_source.uri if first_source else None
            page.dedup_key = first_source.dedup_key if first_source else None
        return pages

    def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID (scans all KBs)."""
        for kb_id, dir_name in self._index.items():
            ctx = self._open_context(kb_id, dir_name)
            source = ctx.db.get_source(source_id)
            if source is not None:
                return source
        return None

    def list_sources(self, kb_id: str) -> list[Source]:
        """List all sources in a KB."""
        ctx = self._ctx(kb_id)
        return ctx.db.list_sources(kb_id)

    def get_links(self, page_id: str, direction: str = "outbound") -> list[WikiPage]:
        """Get pages linked to/from a page."""
        ctx, _ = self._find_context_by_page(page_id)
        return ctx.wiki.get_linked_pages(page_id, direction)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def lint(self, kb_id: str) -> LintReport:
        """Run all lint checks on a KB."""
        ctx = self._ctx(kb_id)
        return ctx.linter.lint(kb_id)

    def lint_fix(self, kb_id: str) -> list[LintIssue]:
        """Auto-fix fixable lint issues in a KB."""
        ctx = self._ctx(kb_id)
        return ctx.linter.auto_fix(kb_id)

    def rebuild_index(self, kb_id: str) -> WikiPage:
        """Regenerate the wiki index page for a KB."""
        ctx = self._ctx(kb_id)
        return ctx.wiki.generate_index(kb_id)

    def export(self, kb_id: str, output_dir: Path) -> list[Path]:
        """Export all wiki pages to markdown files."""
        ctx = self._ctx(kb_id)
        return ctx.exporter.export(kb_id, output_dir)

    # ------------------------------------------------------------------
    # Pipeline Status
    # ------------------------------------------------------------------

    def get_pipeline_status(self, kb_id: str) -> list[PipelineRun]:
        """List all pipeline runs for a KB."""
        dir_name = self._index.get(kb_id)
        if dir_name is None:
            return []
        ctx = self._open_context(kb_id, dir_name)
        return ctx.pipeline.list_runs(kb_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enrich_kb(self, kb: Knowledgebase, ctx: _KBContext) -> Knowledgebase:
        """Add source_count and page_count from DB."""
        kb.source_count = ctx.db.count_sources(kb.id)
        kb.page_count = ctx.db.count_wiki_pages(kb.id)
        return kb
