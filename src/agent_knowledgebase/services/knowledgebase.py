"""Top-level knowledgebase lifecycle service coordinating all sub-services."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent_knowledgebase.config import Settings
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
from agent_knowledgebase.services.embeddings import Embedder, create_embedder
from agent_knowledgebase.services.export import MarkdownExporter
from agent_knowledgebase.services.ingestion import IngestionOrchestrator
from agent_knowledgebase.services.lint import LintIssue, LintReport, WikiLinter
from agent_knowledgebase.services.pipeline import PipelineManager
from agent_knowledgebase.services.query import QueryOrchestrator, SearchResult
from agent_knowledgebase.services.vectorstore import VectorStore, create_vectorstore
from agent_knowledgebase.services.wiki import WikiManager


class KnowledgebaseService:
    """Coordinates all sub-services for full knowledgebase lifecycle.

    This is the primary entry-point for consumers who want to create KBs,
    ingest sources, query, lint, and export -- without needing to manage
    individual sub-services directly.
    """

    def __init__(self, config: Settings) -> None:
        self._config = config.resolve_paths()

        # Core storage
        self._db = Database(self._config.db_path)

        # Embedder (shared across KBs)
        self._embedder: Embedder = create_embedder(self._config)

        # Sub-services that depend only on DB
        self._wiki = WikiManager(self._db)
        self._pipeline = PipelineManager(self._db)
        self._ingestion = IngestionOrchestrator(self._config)
        self._linter = WikiLinter(self._wiki, self._db)
        self._exporter = MarkdownExporter(self._wiki)

        # Per-KB vectorstore cache (collection_name = kb_id)
        self._vectorstores: dict[str, VectorStore] = {}

    # ------------------------------------------------------------------
    # KB Lifecycle
    # ------------------------------------------------------------------

    def create_kb(self, name: str, description: str = "") -> Knowledgebase:
        """Create a new knowledgebase."""
        kb = Knowledgebase(name=name, description=description)
        self._db.insert_knowledgebase(kb)
        return self._enrich_kb(kb)

    def get_kb(self, kb_id: str) -> Knowledgebase | None:
        """Get KB by ID, enriched with source_count and page_count."""
        kb = self._db.get_knowledgebase(kb_id)
        if kb is None:
            return None
        return self._enrich_kb(kb)

    def list_kbs(self) -> list[Knowledgebase]:
        """List all KBs, enriched with counts."""
        return [self._enrich_kb(kb) for kb in self._db.list_knowledgebases()]

    def delete_kb(self, kb_id: str) -> None:
        """Delete a KB and ALL associated data.

        Cascade order respects FK constraints:
        pipeline_runs -> chunks -> wiki_pages -> sources -> KB.
        """
        # Pipeline runs reference sources, so delete first
        self._db.delete_pipeline_runs_by_kb(kb_id)
        # Chunks reference sources
        self._db.delete_chunks_by_kb(kb_id)
        # Wiki pages (handles FTS, links, page_sources internally)
        self._db.delete_wiki_pages_by_kb(kb_id)
        # Sources (also cleans up any remaining page_sources)
        self._db.delete_sources_by_kb(kb_id)
        # Clean up vectorstore
        self._vectorstores.pop(kb_id, None)
        # Finally delete the KB record
        self._db.delete_knowledgebase(kb_id)

    # ------------------------------------------------------------------
    # Source Ingestion (Pipeline-Gated)
    # ------------------------------------------------------------------

    def ingest_source(
        self,
        kb_id: str,
        source_type: SourceType,
        uri: str,
        metadata: dict | None = None,
    ) -> Source:
        """Full ingestion pipeline for a new source.

        Phases: initialize -> read_source -> chunk -> embed ->
        integrate_wiki -> finalize.
        """
        # 1. Verify KB exists
        kb = self._db.get_knowledgebase(kb_id)
        if kb is None:
            raise ValueError(f"KB {kb_id} not found")

        # 2. Create source record
        source = Source(
            kb_id=kb_id,
            source_type=source_type,
            uri=uri,
            metadata=metadata or {},
            status=SourceStatus.ingesting,
        )
        self._db.insert_source(source)

        # 3. Start pipeline
        run = self._pipeline.start_run(kb_id, source.id)

        try:
            # Phase 1: initialize (already running from start_run)
            self._pipeline.complete_phase(run.id)
            self._pipeline.advance_phase(run.id)

            # Phase 2: read_source -- ingest and chunk content
            chunks = self._ingestion.ingest(source_type, uri, metadata)
            for chunk in chunks:
                chunk.source_id = source.id
                chunk.kb_id = kb_id
            self._pipeline.complete_phase(run.id)
            self._pipeline.advance_phase(run.id)

            # Phase 3: chunk (chunking was done in read_source, just advance)
            self._pipeline.complete_phase(run.id)
            self._pipeline.advance_phase(run.id)

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
            # Store chunks in DB
            for chunk in chunks:
                self._db.insert_chunk(chunk)
            self._pipeline.complete_phase(run.id)
            self._pipeline.advance_phase(run.id)

            # Phase 5: integrate_wiki -- create a summary page for this source
            summary_content = f"# {uri}\n\nSource type: {source_type.value}\n\n"
            summary_content += f"Chunks ingested: {len(chunks)}\n"
            if chunks:
                summary_content += (
                    f"\n## Content Preview\n\n{chunks[0].content[:500]}..."
                )
            self._wiki.create_page(
                kb_id=kb_id,
                title=f"Source: {uri}",
                content=summary_content,
                page_type=PageType.summary,
                source_ids=[source.id],
            )
            self._pipeline.complete_phase(run.id)
            self._pipeline.advance_phase(run.id)

            # Phase 6: finalize
            source.status = SourceStatus.ingested
            source.chunk_count = len(chunks)
            source.ingested_at = datetime.now(UTC)
            self._db.update_source(source)
            self._pipeline.complete_phase(run.id)

            return source

        except Exception as e:
            self._pipeline.fail_phase(run.id, str(e))
            source.status = SourceStatus.failed
            self._db.update_source(source)
            raise

    def update_source(self, source_id: str) -> Source:
        """Re-ingest a source (delete old chunks/vectors, re-run pipeline)."""
        source = self._db.get_source(source_id)
        if source is None:
            raise ValueError(f"Source {source_id} not found")

        # Delete old chunks from vectorstore
        old_chunks = self._db.list_chunks(source_id)
        if old_chunks:
            vs = self._get_vectorstore(source.kb_id)
            vs.delete([c.id for c in old_chunks])
        # Delete old chunks from DB
        self._db.delete_chunks_by_source(source_id)

        # Re-ingest
        return self.ingest_source(
            source.kb_id, source.source_type, source.uri, source.metadata
        )

    def remove_source(self, source_id: str) -> None:
        """Remove a source and its chunks/vectors from the KB."""
        source = self._db.get_source(source_id)
        if source is None:
            raise ValueError(f"Source {source_id} not found")

        # Delete chunks from vectorstore
        old_chunks = self._db.list_chunks(source_id)
        if old_chunks:
            vs = self._get_vectorstore(source.kb_id)
            vs.delete([c.id for c in old_chunks])

        # Delete pipeline runs referencing this source
        self._db.delete_pipeline_runs_by_source(source_id)
        # Delete chunks from DB
        self._db.delete_chunks_by_source(source_id)
        # Remove page-source associations referencing this source
        self._db.delete_page_sources_by_source(source_id)
        # Delete the source record
        self._db.delete_source(source_id)

    # ------------------------------------------------------------------
    # Query (No Pipeline Required)
    # ------------------------------------------------------------------

    def query(self, kb_id: str, text: str, top_k: int = 10) -> list[SearchResult]:
        """Semantic query across a KB."""
        vs = self._get_vectorstore(kb_id)
        orchestrator = QueryOrchestrator(vs, self._embedder, self._wiki)
        return orchestrator.query(text, kb_id, top_k=top_k)

    def search(self, kb_id: str, text: str, top_k: int = 10) -> list[SearchResult]:
        """Keyword search across a KB."""
        vs = self._get_vectorstore(kb_id)
        orchestrator = QueryOrchestrator(vs, self._embedder, self._wiki)
        return orchestrator.search(text, kb_id, top_k=top_k)

    def hybrid_query(
        self, kb_id: str, text: str, top_k: int = 10
    ) -> list[SearchResult]:
        """Combined semantic + keyword search."""
        vs = self._get_vectorstore(kb_id)
        orchestrator = QueryOrchestrator(vs, self._embedder, self._wiki)
        return orchestrator.hybrid_query(text, kb_id, top_k=top_k)

    # ------------------------------------------------------------------
    # Wiki Operations
    # ------------------------------------------------------------------

    def get_page(self, page_id: str) -> WikiPage | None:
        """Get a wiki page by ID."""
        return self._wiki.get_page(page_id)

    def list_pages(
        self, kb_id: str, page_type: PageType | None = None
    ) -> list[WikiPage]:
        """List wiki pages in a KB, optionally filtered by type."""
        return self._wiki.list_pages(kb_id, page_type)

    def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID."""
        return self._db.get_source(source_id)

    def list_sources(self, kb_id: str) -> list[Source]:
        """List all sources in a KB."""
        return self._db.list_sources(kb_id)

    def get_links(self, page_id: str, direction: str = "outbound") -> list[WikiPage]:
        """Get pages linked to/from a page.

        Parameters
        ----------
        page_id:
            The page whose links are queried.
        direction:
            ``"outbound"`` (default) or ``"inbound"``.
        """
        return self._wiki.get_linked_pages(page_id, direction)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def lint(self, kb_id: str) -> LintReport:
        """Run all lint checks on a KB."""
        return self._linter.lint(kb_id)

    def lint_fix(self, kb_id: str) -> list[LintIssue]:
        """Auto-fix fixable lint issues in a KB."""
        return self._linter.auto_fix(kb_id)

    def rebuild_index(self, kb_id: str) -> WikiPage:
        """Regenerate the wiki index page for a KB."""
        return self._wiki.generate_index(kb_id)

    def export(self, kb_id: str, output_dir: Path) -> list[Path]:
        """Export all wiki pages to markdown files."""
        return self._exporter.export(kb_id, output_dir)

    # ------------------------------------------------------------------
    # Pipeline Status
    # ------------------------------------------------------------------

    def get_pipeline_status(self, kb_id: str) -> list[PipelineRun]:
        """List all pipeline runs for a KB."""
        return self._pipeline.list_runs(kb_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_vectorstore(self, kb_id: str) -> VectorStore:
        """Get or create a vectorstore collection for a KB."""
        if kb_id not in self._vectorstores:
            self._vectorstores[kb_id] = create_vectorstore(
                self._config, collection_name=kb_id
            )
        return self._vectorstores[kb_id]

    def _enrich_kb(self, kb: Knowledgebase) -> Knowledgebase:
        """Add source_count and page_count from DB."""
        kb.source_count = self._db.count_sources(kb.id)
        kb.page_count = self._db.count_wiki_pages(kb.id)
        return kb
