"""Top-level knowledgebase lifecycle service coordinating all sub-services.

Each knowledgebase is stored in its own subdirectory under
``<saves_dir>/<sanitized-name>/``, containing a per-KB SQLite database
and ChromaDB directory.  A lightweight ``.index.json`` at the
``saves_dir`` level maps ``kb_id -> dir_name`` for fast lookups.

Concurrency model
-----------------
FastMCP runs synchronous tool handlers in a thread pool.  To prevent
races when multiple tool calls target the *same* kb_id concurrently
(e.g., two ``kb_ingest_batch`` calls for the same KB), :meth:`ingest_source`
holds a per-kb_id ``threading.Lock`` for the entire pipeline duration.

This is **single-process-only** serialization — it does not coordinate
across multiple Python processes.  Cross-process isolation (e.g.,
Postgres advisory locks, filesystem ``flock``) is future work.

Multi-process deployments: see ``docs/cross-process-lock-recipe.md``
for opt-in ``filelock`` / ``portalocker`` / Postgres-advisory /
raw-``fcntl`` recipes.  The built-in ``threading.Lock`` does NOT
protect against concurrent ingest from a second Python process.

Why ``threading.Lock`` and not ``asyncio.Lock``?
Because this codebase is entirely synchronous — no ``async def``
anywhere in ``src/agent_knowledgebase/``.  ``asyncio.Lock`` requires a
running event loop and only provides mutual exclusion within a single
event loop, so it would be wrong here.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from agent_knowledgebase.backends import RetrieverBackend, get_backend
from agent_knowledgebase.config import Settings, sanitize_kb_dir_name
from agent_knowledgebase.database import Database
from agent_knowledgebase.models import (
    Knowledgebase,
    PageType,
    PipelineRun,
    RunStatus,
    Source,
    SourceStatus,
    SourceType,
    WikiPage,
)
from agent_knowledgebase.services.dedup_service import DEFAULT_DEDUP_POLICY, DedupPolicy, resolve_dedup_action
from agent_knowledgebase.services.embeddings import (
    Embedder,
    create_embedder,
    create_embedder_for_model,
)
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
from agent_knowledgebase.services.stderr_log import knowledgebase_stderr_log
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

        # Per-kb_id ingestion locks (threading, not asyncio — see module docstring).
        # _kb_locks maps kb_id -> Lock; _kb_locks_guard serialises dict insertion
        # so two threads discovering the same kb_id for the first time don't race.
        self._kb_locks: dict[str, threading.Lock] = {}
        self._kb_locks_guard: threading.Lock = threading.Lock()

        # RetrieverBackend abstraction (Phase 2 redesign).  The factory
        # dispatches on ``self._config.kb_backend`` and returns a real
        # backend wrapping the existing chromadb pipeline (default), or
        # raises NotImplementedError for markdown / lightrag until those
        # phases land.
        #
        # Phase 4 (per-KB routing): the global backend is still
        # constructed here so legacy code paths that don't have a
        # specific kb_id (cross-KB scans like ``get_page``,
        # ``get_source``) keep working unchanged. Per-kb_id resolution
        # routes through :meth:`_backend_for` which consults the
        # ``.migrated_to`` sentinel + ``kb_backend_per_kb`` mapping +
        # global ``kb_backend`` (in that precedence order) and caches
        # the resulting backend per kb_id. spec_id:
        # 70ab2170-381a-4657-bcd1-28a40c6f369b
        self._backend: RetrieverBackend = get_backend(self._config, service=self)
        self._backends_per_kb: dict[str, RetrieverBackend] = {}
        self._backends_per_kb_lock: threading.Lock = threading.Lock()

        # Phase 4 (I-01): per-(kb_id, snapshot-tuple) embedder cache so
        # repeated queries against an unchanged KB don't pay the
        # snapshot read + ``create_embedder_for_model`` build cost on
        # every call. Keyed on (kb_id, model, provider, base_url) and
        # invalidated on (a) per-KB backend cache invalidation
        # (delete_kb / kb_migrate cutover), and (b) every
        # ``_ingest_source_locked`` completion (the embedder identity
        # may have shifted with new chunks). The lock guards both the
        # dict mutation and the cached entries during invalidation.
        self._query_embedder_cache: dict[
            tuple[str, str | None, str | None, str | None], Embedder
        ] = {}
        self._query_embedder_cache_lock: threading.Lock = threading.Lock()

    @property
    def _embedder(self) -> Embedder:
        """Lazily instantiate the embedder on first use."""
        if self._embedder_instance is None:
            self._embedder_instance = create_embedder(self._config)
        return self._embedder_instance

    def _query_embedder_for(self, kb_id: str) -> Embedder:
        """Return the :class:`Embedder` to use when querying *kb_id*.

        Retrieval requires the query vector to come from the same model
        that produced the stored vectors — otherwise similarity scores
        are meaningless (and dimensions may not even match).

        Phase 4 (per-page provider snapshot): the resolution now reads
        the *(model, provider, base_url)* triple stamped on each chunk
        at ingest time via :meth:`Database.get_embedding_snapshot`. The
        full snapshot is passed into
        :func:`create_embedder_for_model` so an existing v0.6.0 KB
        ingested under ``provider=ollama`` /
        ``base_url=http://127.0.0.1:11434`` stays queryable AFTER the
        Phase 5 default flip to ``provider=remote`` /
        ``base_url=...openai.com``. Falls back to the globally
        configured embedder when the KB has no chunks yet or no
        ``embedding_model`` metadata (pre-0.7 ingestions).

        Performance (I-01): the snapshot read uses sqlite-side
        aggregation (single index pass, no Python-side json.loads), and
        the rebuilt embedder is cached per
        (kb_id, model, provider, base_url) so repeated queries against
        an unchanged KB skip the rebuild entirely. The cache is
        invalidated on per-KB backend invalidation (delete_kb /
        kb_migrate) and on each ``_ingest_source_locked`` completion.
        spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
        """
        ctx = self._ctx(kb_id)
        snapshot = ctx.db.get_embedding_snapshot(kb_id)
        if snapshot is None:
            return self._embedder
        dominant_model, dominant_provider, dominant_base_url = snapshot
        if dominant_model is None:
            return self._embedder
        # When the snapshot's tuple matches the currently-configured
        # embedder we can return the cached instance and skip a fresh
        # build; otherwise rebuild via the snapshot so the right
        # provider+base_url is wired in.
        if (
            dominant_model == self._embedder.model_name
            and (
                dominant_provider is None
                or dominant_provider == self._config.embedding_provider
            )
            and (
                dominant_base_url is None
                or dominant_base_url == self._config.embed_base_url
            )
        ):
            return self._embedder
        # Snapshot diverges from the configured embedder; consult the
        # per-(kb_id, snapshot-tuple) cache before rebuilding. The
        # cache key includes kb_id so two KBs that share a
        # (model, provider, base_url) tuple still get distinct cache
        # entries — the embedder instance is the same shape, but
        # invalidation is scoped per-KB (an ingest into KB A shouldn't
        # invalidate KB B's cached entry).
        cache_key = (
            kb_id,
            dominant_model,
            dominant_provider,
            dominant_base_url,
        )
        cached = self._query_embedder_cache.get(cache_key)
        if cached is not None:
            return cached
        with self._query_embedder_cache_lock:
            cached = self._query_embedder_cache.get(cache_key)
            if cached is not None:
                return cached
            embedder = create_embedder_for_model(
                self._config,
                dominant_model,
                provider=dominant_provider,
                base_url=dominant_base_url,
            )
            self._query_embedder_cache[cache_key] = embedder
            return embedder

    def _invalidate_query_embedder_cache(self, kb_id: str | None = None) -> None:
        """Drop cached query embedders for *kb_id* (or all if None).

        Called on per-KB backend invalidation and on each successful
        ingest, since either event can shift the dominant snapshot
        tuple for the KB.
        """
        with self._query_embedder_cache_lock:
            if kb_id is None:
                self._query_embedder_cache.clear()
            else:
                stale = [k for k in self._query_embedder_cache if k[0] == kb_id]
                for key in stale:
                    self._query_embedder_cache.pop(key, None)

    def _backend_for(self, kb_id: str) -> RetrieverBackend:
        """Return the :class:`RetrieverBackend` for *kb_id*.

        Phase 4 routing: cached per kb_id; resolution precedence is
        sentinel file > ``kb_backend_per_kb`` mapping > global
        ``kb_backend``. Cache invalidation happens automatically when
        ``delete_kb`` runs (see :meth:`_invalidate_backend_cache`).
        """
        backend = self._backends_per_kb.get(kb_id)
        if backend is not None:
            return backend
        with self._backends_per_kb_lock:
            backend = self._backends_per_kb.get(kb_id)
            if backend is not None:
                return backend
            resolved = get_backend(self._config, service=self, kb_id=kb_id)
            self._backends_per_kb[kb_id] = resolved
            return resolved

    def _invalidate_backend_cache(self, kb_id: str | None = None) -> None:
        """Drop cached per-KB backend(s) so the next call re-resolves.

        When *kb_id* is ``None`` the entire cache is cleared (used after
        a config-level change like a ``kb_migrate`` that may flip
        routing for the current process). When a kb_id is supplied,
        only that entry is dropped.

        Also drops the per-KB query embedder cache (I-01) since the
        snapshot tuple may have changed in lockstep with the backend
        flip (e.g. a markdown -> chromadb cutover repopulates the
        chunks table with a fresh provider snapshot).
        """
        with self._backends_per_kb_lock:
            if kb_id is None:
                self._backends_per_kb.clear()
            else:
                self._backends_per_kb.pop(kb_id, None)
        self._invalidate_query_embedder_cache(kb_id)

    def _get_kb_lock(self, kb_id: str) -> threading.Lock:
        """Return (creating lazily) the per-kb_id ingestion lock.

        The guard lock ensures that two threads discovering the same
        kb_id simultaneously both see the *same* Lock object and don't
        accidentally create two separate locks for the same kb_id.
        """
        # Fast path: lock already exists (no guard needed).
        lock = self._kb_locks.get(kb_id)
        if lock is not None:
            return lock
        # Slow path: atomically insert via guard.
        with self._kb_locks_guard:
            # Re-check under the guard to handle the race where two threads
            # both passed the fast-path check simultaneously.
            return self._kb_locks.setdefault(kb_id, threading.Lock())

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

        # Release the per-kb_id lock entry so it doesn't drift in long-running
        # processes.  Safe to pop even if a worker still holds the lock object —
        # the holder still owns the object; a future re-create will get a fresh one.
        with self._kb_locks_guard:
            self._kb_locks.pop(kb_id, None)

        # Phase 4: drop any cached backend for this kb_id so a future
        # re-create with the same kb_id gets a fresh backend instance
        # (and the new directory layout the cached backend may have
        # been pointing at is not silently shared).
        self._invalidate_backend_cache(kb_id)

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
        request_id: str | None = None,
        tool_caller_version: str | None = None,
        batch_size: int | None = None,
        explicit_backend: RetrieverBackend | None = None,
    ) -> Source:
        """Full ingestion pipeline for a new source.

        Concurrent calls targeting the *same* kb_id are serialised by a
        per-kb_id ``threading.Lock``.  Calls against *different* kb_ids
        run concurrently — there is no global lock.

        This is single-process serialisation only.  Cross-process
        isolation (e.g., Postgres advisory locks, filesystem flock) is
        future work.

        ``request_id`` / ``tool_caller_version`` / ``batch_size`` are
        optional caller correlators threaded through to the
        PipelineRun telemetry row and the structured stderr log.

        ``explicit_backend`` is a Phase-4-migration-only escape hatch:
        when supplied, the index write skips the per-KB cache lookup
        (``_backend_for(kb_id)``) and routes directly to the supplied
        backend instance. This is required by the markdown -> chromadb
        reverse migration path because the ``.migrated_to`` sentinel
        still says ``markdown`` while the import is in flight, so the
        per-KB cache would otherwise route writes back into the wiki
        and ZERO vectors would land in chromadb.
        """
        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="lock_acquire",
            phase="lock",
            elapsed_ms=0,
            rows_in=0,
            rows_ok=0,
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=request_id,
            tool_caller_version=tool_caller_version,
        )
        lock_acquired_at = time.monotonic()
        with self._get_kb_lock(kb_id):
            knowledgebase_stderr_log(
                kb_id=kb_id,
                op="lock_acquired",
                phase="lock",
                elapsed_ms=0,
                rows_in=0,
                rows_ok=0,
                rows_skipped=0,
                rows_failed=0,
                dedup_policy="n/a",
                request_id=request_id,
                tool_caller_version=tool_caller_version,
            )
            try:
                return self._ingest_source_locked(
                    kb_id,
                    source_type,
                    uri,
                    metadata,
                    dedup_key,
                    dedup_policy,
                    request_id=request_id,
                    tool_caller_version=tool_caller_version,
                    batch_size=batch_size,
                    explicit_backend=explicit_backend,
                )
            finally:
                elapsed_ms = int((time.monotonic() - lock_acquired_at) * 1000)
                knowledgebase_stderr_log(
                    kb_id=kb_id,
                    op="lock_release",
                    phase="lock",
                    elapsed_ms=elapsed_ms,
                    rows_in=0,
                    rows_ok=0,
                    rows_skipped=0,
                    rows_failed=0,
                    dedup_policy="n/a",
                    request_id=request_id,
                    tool_caller_version=tool_caller_version,
                )

    def _ingest_source_locked(
        self,
        kb_id: str,
        source_type: SourceType,
        uri: str,
        metadata: dict | None = None,
        dedup_key: str | None = None,
        dedup_policy: DedupPolicy = DEFAULT_DEDUP_POLICY,
        request_id: str | None = None,
        tool_caller_version: str | None = None,
        batch_size: int | None = None,
        explicit_backend: RetrieverBackend | None = None,
    ) -> Source:
        """Inner pipeline body — called only while the kb_id lock is held.

        ``request_id`` / ``tool_caller_version`` / ``batch_size`` are
        captured on the PipelineRun v0.6.0 telemetry row along with the
        resolved dedup counters (``ingested`` / ``skipped`` / ``replaced``
        / ``failed``) and the ``ended_at`` wall-clock timestamp.

        A telemetry row is written for every outcome including ``skip``
        so callers can see the skip via ``kb_pipeline_status``.

        ``explicit_backend`` (Phase-4-migration-only): when supplied,
        the index write routes through this backend instead of the
        cached per-KB backend. Required by ``import_from_markdown`` —
        the ``.migrated_to`` sentinel still says ``markdown`` while the
        reverse migration is in flight, so the cached backend is the
        wrong target. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
        """
        ctx = self._ctx(kb_id)

        # Verify KB exists
        kb = ctx.db.get_knowledgebase(kb_id)
        if kb is None:
            raise ValueError(f"KB {kb_id} not found")

        policy_value = (
            dedup_policy.value if isinstance(dedup_policy, DedupPolicy) else str(dedup_policy)
        )

        # Dedup check
        action, existing = resolve_dedup_action(ctx.db, kb_id, dedup_key, dedup_policy)

        # Skip outcome: emit a telemetry row and return the existing source.
        if action == "skip":
            skip_run = ctx.pipeline.start_run(kb_id, existing.id if existing else None)  # type: ignore[union-attr]
            skip_run.status = RunStatus.completed
            skip_run.completed_at = datetime.now(UTC)
            skip_run.ended_at = skip_run.completed_at
            skip_run.ingested = 0
            skip_run.skipped = 1
            skip_run.replaced = 0
            skip_run.failed = 0
            skip_run.batch_size = batch_size
            skip_run.dedup_policy = policy_value
            skip_run.request_id = request_id
            skip_run.tool_caller_version = tool_caller_version
            ctx.db.update_pipeline_run(skip_run)
            return existing  # type: ignore[return-value]

        replaced_count = 0
        if action == "replace":
            self.remove_source(existing.id)  # type: ignore[union-attr]
            replaced_count = 1

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
        # Stamp the telemetry-row capture fields early so they survive
        # a mid-pipeline failure.
        run.batch_size = batch_size
        run.dedup_policy = policy_value
        run.request_id = request_id
        run.tool_caller_version = tool_caller_version
        run.replaced = replaced_count
        ctx.db.update_pipeline_run(run)

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
            # Stamp per-chunk metadata that the retrieval path depends on:
            # ``embedding_model`` drives per-KB embedder selection on query,
            # and ``kb_id`` satisfies the vectorstore's where-filter (which
            # would otherwise return zero results even though the per-KB
            # collection holds the right chunks).
            #
            # Phase 4 (per-page provider snapshot): also stamp
            # ``embedding_provider`` and ``embed_base_url`` so the
            # retrieval path can faithfully rebuild the original
            # embedder via ``create_embedder_for_model(model, provider=...,
            # base_url=...)`` even AFTER the Phase 5 default flip moves
            # ``Settings.embedding_provider`` to a different value. The
            # secret (``embed_api_key``) is intentionally NOT stamped —
            # secrets stay in Settings/env per the FORBIDDEN_KEYS
            # discipline; the snapshot only carries non-secret routing
            # metadata. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
            for chunk in chunks:
                chunk.metadata["embedding_model"] = self._embedder.model_name
                chunk.metadata["kb_id"] = kb_id
                chunk.metadata["embedding_provider"] = self._config.embedding_provider
                if self._config.embed_base_url is not None:
                    chunk.metadata["embed_base_url"] = self._config.embed_base_url
            # Route the index write through the RetrieverBackend
            # abstraction (Phase 2). For chromadb this is a thin wrapper
            # around ``vectorstore.add(...)``; markdown (Phase 3) /
            # lightrag (Phase 6) implementations will diverge here.
            #
            # Phase 4: per-KB routing via ``_backend_for(kb_id)`` so
            # that ``AGENT_KB_BACKEND_PER_KB`` overrides + the
            # ``.migrated_to`` sentinel both take effect. Falls through
            # to the global default when neither is set, preserving
            # Phase 2/3 behavior.
            if chunks:
                # Phase 4: when an explicit backend is supplied (e.g.
                # the markdown -> chromadb reverse migration), bypass
                # the per-KB cache so writes don't get re-routed back
                # into the OLD backend by the still-valid sentinel.
                target_backend = (
                    explicit_backend
                    if explicit_backend is not None
                    else self._backend_for(kb_id)
                )
                target_backend.index(
                    kb_id=kb_id,
                    documents=[
                        {
                            "id": c.id,
                            "content": text,
                            "metadata": c.metadata,
                            "embedding": emb,
                        }
                        for c, text, emb in zip(chunks, texts, embeddings)
                    ],
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

            # I-01: drop the cached query embedder for this KB so the
            # next query re-reads the snapshot. The new chunks may have
            # shifted the dominant (model, provider, base_url) tuple
            # (e.g. switching to a new embedder mid-KB) and a stale
            # cache entry would silently return the old embedder.
            self._invalidate_query_embedder_cache(kb_id)

            # v0.6.0 telemetry row — final counters & ended_at.
            final_run = ctx.pipeline.get_run(run.id)
            if final_run is not None:
                final_run.ended_at = datetime.now(UTC)
                final_run.ingested = 1
                final_run.skipped = 0
                final_run.replaced = replaced_count
                final_run.failed = 0
                final_run.batch_size = batch_size
                final_run.dedup_policy = policy_value
                final_run.request_id = request_id
                final_run.tool_caller_version = tool_caller_version
                ctx.db.update_pipeline_run(final_run)

            return source

        except Exception as e:
            ctx.pipeline.fail_phase(run.id, str(e))
            source.status = SourceStatus.failed
            ctx.db.update_source(source)
            # v0.6.0 telemetry row — record failure counters.
            fail_run = ctx.pipeline.get_run(run.id)
            if fail_run is not None:
                fail_run.ended_at = datetime.now(UTC)
                fail_run.ingested = 0
                fail_run.skipped = 0
                fail_run.replaced = replaced_count
                fail_run.failed = 1
                fail_run.batch_size = batch_size
                fail_run.dedup_policy = policy_value
                fail_run.request_id = request_id
                fail_run.tool_caller_version = tool_caller_version
                ctx.db.update_pipeline_run(fail_run)
            raise

    def update_source(self, source_id: str) -> Source:
        """Re-ingest a source (delete old chunks/vectors, re-run pipeline).

        The entire operation — cleanup *and* re-ingest — is serialised under the
        per-kb_id lock so no concurrent ``ingest_source`` call can observe the
        torn state between "old chunks deleted" and "new chunks inserted".
        ``_ingest_source_locked`` is called directly (bypassing the lock
        acquisition in ``ingest_source``) to avoid a self-deadlock, since
        ``threading.Lock`` is non-reentrant.

        Note: when the dedup policy resolves to ``'replace'``, the inner
        ``_ingest_source_locked`` path calls ``self.remove_source(existing.id)``
        which itself routes through ``self._backend.delete`` — so all delete
        paths are backend-routed.
        """
        ctx, _ = self._find_context_by_source(source_id)
        source = ctx.db.get_source(source_id)
        if source is None:
            raise ValueError(f"Source {source_id} not found")

        kb_id = source.kb_id
        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="lock_acquire",
            phase="lock",
            elapsed_ms=0,
            rows_in=0,
            rows_ok=0,
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=None,
            tool_caller_version=None,
        )
        lock_acquired_at = time.monotonic()
        with self._get_kb_lock(kb_id):
            knowledgebase_stderr_log(
                kb_id=kb_id,
                op="lock_acquired",
                phase="lock",
                elapsed_ms=0,
                rows_in=0,
                rows_ok=0,
                rows_skipped=0,
                rows_failed=0,
                dedup_policy="n/a",
                request_id=None,
                tool_caller_version=None,
            )
            try:
                # Delete old chunks from the backend.  Routes through
                # ``self._backend_for(kb_id).delete`` (Phase 4 per-KB
                # routing) so the markdown / lightrag backends can
                # override the deletion path without touching this
                # service.
                old_chunks = ctx.db.list_chunks(source_id)
                if old_chunks:
                    self._backend_for(kb_id).delete(
                        kb_id=kb_id, ids=[c.id for c in old_chunks]
                    )
                ctx.db.delete_chunks_by_source(source_id)

                return self._ingest_source_locked(
                    kb_id, source.source_type, source.uri, source.metadata
                )
            finally:
                elapsed_ms = int((time.monotonic() - lock_acquired_at) * 1000)
                knowledgebase_stderr_log(
                    kb_id=kb_id,
                    op="lock_release",
                    phase="lock",
                    elapsed_ms=elapsed_ms,
                    rows_in=0,
                    rows_ok=0,
                    rows_skipped=0,
                    rows_failed=0,
                    dedup_policy="n/a",
                    request_id=None,
                    tool_caller_version=None,
                )

    def remove_source(self, source_id: str) -> None:
        """Remove a source and its chunks/vectors from the KB.

        The vector deletion routes through the
        :class:`~agent_knowledgebase.backends.RetrieverBackend`
        abstraction so future backends (markdown, lightrag) can swap
        their own deletion semantics in. SQL-level cleanup
        (pipeline_runs, chunks, page_sources, source rows) stays here
        since those tables are backend-agnostic.
        """
        ctx, _ = self._find_context_by_source(source_id)
        source = ctx.db.get_source(source_id)
        if source is None:
            raise ValueError(f"Source {source_id} not found")

        old_chunks = ctx.db.list_chunks(source_id)
        if old_chunks:
            self._backend_for(source.kb_id).delete(
                kb_id=source.kb_id, ids=[c.id for c in old_chunks]
            )

        ctx.db.delete_pipeline_runs_by_source(source_id)
        ctx.db.delete_chunks_by_source(source_id)
        ctx.db.delete_page_sources_by_source(source_id)
        ctx.db.delete_source(source_id)

    # ------------------------------------------------------------------
    # Query (No Pipeline Required)
    # ------------------------------------------------------------------

    def query(self, kb_id: str, text: str, top_k: int | None = None) -> list[SearchResult]:
        """Semantic query across a KB.

        Routes through ``self._backend_for(kb_id).query`` (Phase 4
        per-KB routing — sentinel/per-kb mapping/global default in that
        precedence). The chromadb default returns dicts shaped like
        :class:`SearchResult`'s ``__dict__`` so we round-trip them back
        into :class:`SearchResult` instances to preserve the v0.6.0
        return type.
        """
        if top_k is None:
            top_k = default_top_k(self._config)
        # Resolve the kb context here so a missing kb_id raises the
        # historical ValueError before the backend is consulted; the
        # backend would also raise but with a less specific message.
        self._ctx(kb_id)
        rows = self._backend_for(kb_id).query(kb_id=kb_id, text=text, top_k=top_k)
        return [SearchResult(**row) for row in rows]

    def search(self, kb_id: str, text: str, top_k: int | None = None) -> list[SearchResult]:
        """Keyword search across a KB.

        Routes through ``self._backend_for(kb_id).search`` (Phase 4
        per-KB routing). See :meth:`query` for the round-trip rationale.
        """
        if top_k is None:
            top_k = default_top_k(self._config)
        self._ctx(kb_id)
        rows = self._backend_for(kb_id).search(kb_id=kb_id, text=text, top_k=top_k)
        return [SearchResult(**row) for row in rows]

    def hybrid_query(self, kb_id: str, text: str, top_k: int | None = None) -> list[SearchResult]:
        """Combined semantic + keyword search.

        The hybrid composition itself stays in
        :class:`QueryOrchestrator` since the merge / weighting logic is
        backend-agnostic; the per-leg vector + FTS calls are still the
        v0.6.0 chromadb path because ``self._backend`` defaults to
        chromadb and ``QueryOrchestrator`` reads through the same
        helpers the backend wraps. Phase 3 may revisit if the markdown
        backend ships a different hybrid story.
        """
        if top_k is None:
            top_k = default_top_k(self._config)
        vector_weight, fts_weight, fetch_mult = hybrid_weights_from(self._config)
        ctx = self._ctx(kb_id)
        vs = self._get_vectorstore(kb_id)
        embedder = self._query_embedder_for(kb_id)
        orchestrator = QueryOrchestrator(vs, embedder, ctx.wiki)
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
    # Migration (Phase 4 — kb_migrate MCP tool entry point)
    # ------------------------------------------------------------------

    def migrate(self, *, kb_id: str, target_backend: str) -> dict:
        """Cut over a KB to a different :class:`RetrieverBackend`.

        Thin pass-through to :func:`services.migration.migrate` so the
        MCP tool layer (``server.py::kb_migrate``) doesn't have to
        import the migration module directly. Returns the
        :meth:`MigrationResult.to_payload` dict for JSON serialization.
        spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
        """
        # Lazy import to keep server.py import time light — see the
        # subagent brief's "Don't import migration.py at server.py
        # import time" note.
        from agent_knowledgebase.services.migration import migrate as _migrate

        result = _migrate(self, kb_id=kb_id, target_backend=target_backend)
        return result.to_payload()

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
        """Add source_count, page_count, and embedding model stats from DB."""
        kb.source_count = ctx.db.count_sources(kb.id)
        kb.page_count = ctx.db.count_wiki_pages(kb.id)
        model_counts = ctx.db.count_chunks_by_embedding_model(kb.id)
        kb.embedding_model_counts = model_counts
        if model_counts:
            kb.dominant_embedding_model = max(model_counts, key=lambda m: model_counts[m])
        else:
            kb.dominant_embedding_model = None
        return kb
