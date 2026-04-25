"""MCP server entry point for agent-knowledgebase."""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

from mcp.server.fastmcp import FastMCP

from pydantic import ValidationError
from pydantic_core import PydanticUndefined

from agent_knowledgebase.config import Settings
from agent_knowledgebase.config_files import (
    DOT_TO_FLAT,
    FORBIDDEN_KEYS,
    NestedJsonConfigSettingsSource,
    load_settings,
    resolve_project_config_path,
    resolve_user_config_path,
    write_candidate_and_validate,
)
from agent_knowledgebase.models import PageType, SourceType
from agent_knowledgebase.services.embeddings import EmbedderUnavailableError
from agent_knowledgebase.services.kb_ingest_service import (
    normalize_dedup_policy,
    run_ingest_batch,
    validate_batch_size,
)
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService

mcp = FastMCP(
    "agent-knowledgebase",
    instructions="Personal knowledgebase plugin — ingest sources, build wiki artifacts, query with vector search",
)

# ---------------------------------------------------------------------------
# Lazy initialization of the service singleton
# ---------------------------------------------------------------------------

_service: KnowledgebaseService | object | None = None


def _config_missing_payload(
    *, missing: str, detail: str
) -> dict[str, str]:
    """Return the structured ``config_missing`` JSON payload."""
    return {
        "error": "config_missing",
        "missing": missing,
        "detail": detail,
    }


def _classify_config_error(exc: BaseException) -> dict[str, str]:
    """Translate a Settings/resolve_paths exception into a structured payload.

    Inspects ``pydantic.ValidationError`` errors to identify the
    specific missing field (typically ``saves_dir`` →
    ``AGENT_KB_SAVES_DIR``) and falls back to an
    ``AGENT_KB_SAVES_DIR``-flavoured payload for path-existence errors
    raised by ``resolve_paths``.
    """
    if isinstance(exc, ValidationError):
        missing_field = "AGENT_KB_SAVES_DIR"
        try:
            errors = exc.errors()
        except Exception:  # noqa: BLE001 — defensive
            errors = []
        for err in errors:
            loc = err.get("loc") or ()
            if not loc:
                continue
            field = str(loc[0])
            # Map Settings field name to AGENT_KB_-prefixed env var.
            missing_field = f"AGENT_KB_{field.upper()}"
            break
        return _config_missing_payload(
            missing=missing_field,
            detail=f"Settings validation failed: {exc}",
        )
    if isinstance(exc, FileNotFoundError):
        return _config_missing_payload(
            missing="AGENT_KB_SAVES_DIR",
            detail=str(exc),
        )
    if isinstance(exc, NotADirectoryError):
        return _config_missing_payload(
            missing="AGENT_KB_SAVES_DIR",
            detail=str(exc),
        )
    return _config_missing_payload(
        missing="unknown",
        detail=f"{type(exc).__name__}: {exc}",
    )


class _ConfigMissingError(Exception):
    """Internal sentinel raised by ``_get_service`` when config is missing.

    Carries the structured ``config_missing`` payload so the
    ``_with_tool_timeout`` wrapper can return it as the tool response
    without each individual ``kb_*`` tool needing to know about the
    failure mode.
    """

    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload
        super().__init__(payload.get("detail", "config missing"))


class _ConfigMissingService:
    """Sentinel service returned by ``_get_service`` when config is missing.

    Every attribute access returns a callable that, when invoked,
    raises :class:`_ConfigMissingError` carrying the structured
    payload. The ``_with_tool_timeout`` wrapper catches that exception
    and returns the payload as the tool's JSON response, so every
    ``kb_*`` tool surfaces a structured ``config_missing`` payload
    instead of an opaque MCP InternalError.

    Why raise instead of return: most ``kb_*`` tools post-process the
    service return value (``_serialize_model_list``, etc.), so a bare
    JSON string returned from ``svc.list_kbs()`` would itself break.
    Raising lets the wrapper short-circuit cleanly at one chokepoint.
    """

    def __init__(self, payload: dict[str, str]) -> None:
        self._payload = dict(payload)

    @property
    def payload(self) -> dict[str, str]:
        return dict(self._payload)

    def __getattr__(self, name: str) -> Callable[..., str]:
        payload = self._payload

        def _raise_config_missing(*_args: object, **_kwargs: object) -> str:
            raise _ConfigMissingError(payload)

        _raise_config_missing.__name__ = f"_config_missing_{name}"
        return _raise_config_missing


def _get_service() -> KnowledgebaseService | _ConfigMissingService:
    global _service
    if _service is None:
        try:
            config = Settings().resolve_paths()
        except (ValidationError, FileNotFoundError, NotADirectoryError) as exc:
            payload = _classify_config_error(exc)
            _service = _ConfigMissingService(payload)
        else:
            _service = KnowledgebaseService(config)
    return _service  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _serialize_dataclass_list(items: list) -> str:
    """Serialize a list of dataclass instances to a JSON array string."""
    return json.dumps([asdict(item) for item in items], default=str)


def _serialize_dataclass(item: object) -> str:
    """Serialize a single dataclass instance to a JSON string."""
    return json.dumps(asdict(item), default=str)  # type: ignore[call-overload]


def _serialize_model_list(items: list) -> str:
    """Serialize a list of Pydantic model instances to a JSON array string."""
    return "[" + ", ".join(item.model_dump_json() for item in items) + "]"


# ---------------------------------------------------------------------------
# Wall-clock tool timeout
# ---------------------------------------------------------------------------
# A stalled embedder, ingestor, or DB call must not block the MCP RPC
# forever: instead we enforce a per-tool wall-clock bound by running each
# call on a dedicated worker thread and timing out via
# ``future.result(timeout=...)``.  On timeout we abandon the worker
# (spawning a fresh one for the next call) and also clear the cached
# ``_service`` so the next call rebuilds SQLite connections in the new
# thread — sqlite3 connections are thread-affine by default, so reusing
# the old service from a new worker would ``ProgrammingError``.

_TOOL_TIMEOUT_SECONDS: float = 180.0  # 3 minutes

_tool_executor: concurrent.futures.ThreadPoolExecutor | None = None
_tool_executor_lock = threading.Lock()
# Per-thread flag: True when we are already running inside the tool
# executor's worker thread.  Used to bypass the executor on re-entrant
# calls (e.g., kb_config_set → kb_config_get) so the inner call does not
# queue behind the outer one on the single-worker executor.
_in_tool_worker = threading.local()

_F = TypeVar("_F", bound=Callable[..., str])


def _get_tool_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _tool_executor
    with _tool_executor_lock:
        if _tool_executor is None:
            _tool_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kb_tool"
            )
        return _tool_executor


def _reset_tool_executor() -> None:
    """Abandon the stuck worker and clear the cached service."""
    global _tool_executor, _service
    with _tool_executor_lock:
        old = _tool_executor
        _tool_executor = None
    if old is not None:
        old.shutdown(wait=False)
    _service = None


def _with_tool_timeout(func: _F) -> _F:
    """Wrap a tool with a wall-clock timeout and structured failure payload.

    On timeout, returns a JSON string
    ``{"error": "tool_timeout", "tool": ..., "timeout_seconds": ...}`` so
    MCP sees a normal tool response instead of a hung RPC.

    Also catches :class:`_ConfigMissingError` raised by the
    ``_ConfigMissingService`` sentinel (when ``_get_service`` cannot
    construct a real service due to missing or invalid configuration)
    and returns the structured ``config_missing`` JSON payload as the
    tool response. This converts the previous opaque MCP InternalError
    into a useful diagnostic for the caller — the Phase 0 Bug-2 fix.
    """
    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> str:
        # Re-entrant call from inside the worker thread: skip the
        # executor hop to avoid self-deadlock on the single worker.
        if getattr(_in_tool_worker, "active", False):
            try:
                return func(*args, **kwargs)
            except _ConfigMissingError as exc:
                return json.dumps(exc.payload)

        def _run(*a: object, **kw: object) -> str:
            _in_tool_worker.active = True
            try:
                return func(*a, **kw)
            finally:
                _in_tool_worker.active = False

        executor = _get_tool_executor()
        future = executor.submit(_run, *args, **kwargs)
        try:
            return future.result(timeout=_TOOL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            _reset_tool_executor()
            return json.dumps({
                "error": "tool_timeout",
                "tool": func.__name__,
                "timeout_seconds": _TOOL_TIMEOUT_SECONDS,
            })
        except _ConfigMissingError as exc:
            return json.dumps(exc.payload)

    return wrapper  # type: ignore[return-value]


# ===================================================================
# Write-Path Tools (Pipeline-Gated)
# ===================================================================


@mcp.tool()
@_with_tool_timeout
def kb_create(name: str, description: str = "") -> str:
    """Create a new knowledgebase.

    Parameters:
        name: Name of the knowledgebase.
        description: Optional description.

    Returns a JSON object of the created knowledgebase.
    """
    svc = _get_service()
    kb = svc.create_kb(name, description)
    return kb.model_dump_json()


@mcp.tool()
@_with_tool_timeout
def kb_delete(kb_id: str) -> str:
    """Delete a knowledgebase and all its data (sources, pages, chunks, vectors).

    Parameters:
        kb_id: ID of the knowledgebase to delete.

    Returns a confirmation message.
    """
    svc = _get_service()
    svc.delete_kb(kb_id)
    return json.dumps({"status": "deleted", "kb_id": kb_id})


@mcp.tool()
@_with_tool_timeout
def kb_ingest(
    kb_id: str,
    source_type: str,
    uri: str,
    metadata: str = "{}",
    dedup_key: str | None = None,
    dedup_policy: str = "skip",
) -> str:
    """Ingest a source into a knowledgebase.

    Parameters:
        kb_id: ID of the target knowledgebase.
        source_type: Type of source (file, directory, codebase, website, sql_database, git_history, api_endpoint).
        uri: Location or path of the source.
        metadata: JSON string of additional metadata.
        dedup_key: Optional stable identifier. When provided, dedup_policy controls behaviour on collision.
        dedup_policy: One of "skip" (default), "replace", "force-add". Ignored when dedup_key is absent.

    Returns a JSON object of the created (or existing) source.
    """
    svc = _get_service()
    st = SourceType(source_type)
    meta = json.loads(metadata)
    policy = normalize_dedup_policy(dedup_policy)
    source = svc.ingest_source(kb_id, st, uri, meta, dedup_key=dedup_key, dedup_policy=policy)
    return source.model_dump_json()


@mcp.tool()
@_with_tool_timeout
def kb_ingest_batch(kb_id: str, sources: str, dedup_policy: str = "skip") -> str:
    """Batch ingest multiple sources into a knowledgebase.

    Parameters:
        kb_id: ID of the target knowledgebase.
        sources: JSON array of objects, each with keys: source_type, uri, metadata (optional),
            dedup_key (optional), dedup_policy (optional per-row override).
        dedup_policy: Top-level default policy — "skip" (default), "replace", or "force-add".
            Each row in sources may carry its own dedup_policy to override this default.

    Returns a JSON array of created (or existing) source objects.

    Raises:
        BatchSizeExceededError: If more than 50 rows with
            ``source_type='sql_database'`` are submitted in a single
            call.  No source record is created and no ingestor runs
            when this error is raised.
    """
    source_defs = json.loads(sources)
    # Hard-reject oversized sql_database batches BEFORE any DB write or
    # ingestor dispatch -- the release-1 guardrail mandated by the
    # round-3 debate synthesizer (overturning the spec's soft-warn).
    validate_batch_size(source_defs)

    default_policy = normalize_dedup_policy(dedup_policy)
    svc = _get_service()
    # `request_id` / `tool_caller_version` are not surfaced on this MCP
    # tool's signature (stable contract) — the batch runner synthesizes
    # a request_id internally so the v0.6.0 telemetry row is populated.
    results = run_ingest_batch(
        svc,
        kb_id,
        source_defs,
        default_policy=default_policy,
    )
    return _serialize_model_list(results)


@mcp.tool()
@_with_tool_timeout
def kb_update_source(source_id: str) -> str:
    """Re-ingest a source (delete old chunks and re-run the ingestion pipeline).

    Parameters:
        source_id: ID of the source to update.

    Returns a JSON object of the updated source.
    """
    svc = _get_service()
    source = svc.update_source(source_id)
    return source.model_dump_json()


@mcp.tool()
@_with_tool_timeout
def kb_remove_source(source_id: str) -> str:
    """Remove a source and all its chunks from the knowledgebase.

    Parameters:
        source_id: ID of the source to remove.

    Returns a confirmation message.
    """
    svc = _get_service()
    svc.remove_source(source_id)
    return json.dumps({"status": "removed", "source_id": source_id})


@mcp.tool()
@_with_tool_timeout
def kb_lint(kb_id: str) -> str:
    """Run wiki health checks on a knowledgebase.

    Checks for orphan pages, missing wikilink targets, stale source references,
    coverage gaps, and index drift.

    Parameters:
        kb_id: ID of the knowledgebase to lint.

    Returns a JSON lint report.
    """
    svc = _get_service()
    report = svc.lint(kb_id)
    return _serialize_dataclass(report)


@mcp.tool()
@_with_tool_timeout
def kb_lint_fix(kb_id: str) -> str:
    """Auto-fix fixable lint issues in a knowledgebase.

    Parameters:
        kb_id: ID of the knowledgebase to fix.

    Returns a JSON array of fixed issues.
    """
    svc = _get_service()
    fixed = svc.lint_fix(kb_id)
    return _serialize_dataclass_list(fixed)


@mcp.tool()
@_with_tool_timeout
def kb_rebuild_index(kb_id: str) -> str:
    """Rebuild the wiki index page for a knowledgebase.

    Parameters:
        kb_id: ID of the knowledgebase.

    Returns a JSON object of the index page.
    """
    svc = _get_service()
    page = svc.rebuild_index(kb_id)
    return page.model_dump_json()


@mcp.tool()
@_with_tool_timeout
def kb_export(kb_id: str, output_dir: str) -> str:
    """Export all wiki pages to markdown files.

    Parameters:
        kb_id: ID of the knowledgebase.
        output_dir: Directory path where files will be written.

    Returns a JSON array of exported file paths.
    """
    svc = _get_service()
    paths = svc.export(kb_id, Path(output_dir))
    return json.dumps([str(p) for p in paths])


@mcp.tool()
@_with_tool_timeout
def kb_migrate(kb_id: str, target_backend: str) -> str:
    """Migrate a knowledgebase between retrieval backends (Phase 4).

    Cuts over the storage layout for *kb_id* to *target_backend*
    (one of ``"chromadb"`` or ``"markdown"``) by:

    1. Materialising the data on disk in the new backend's layout
       (``<saves_dir>/<kb-name>/wiki/`` for markdown,
       ``<saves_dir>/<kb-name>/chroma/`` for chromadb).
    2. Writing the routing sentinel file
       ``<saves_dir>/<kb-name>/.migrated_to`` with the target backend
       name so future ``kb_query`` / ``kb_search`` calls route there.
    3. Leaving the OLD backend's data untouched — rollback is
       deleting the sentinel file. Operators reclaim disk space
       manually.

    Parameters:
        kb_id: ID of the knowledgebase to migrate.
        target_backend: Target backend name; one of ``"chromadb"`` or
            ``"markdown"``. ``"lightrag"`` is rejected (Phase 6 deferred).

    Returns a JSON object summarising the migration:
    ``{"kb_id":..., "target_backend":..., "pages_migrated":N,
    "sentinel_path":..., "notes":[...], "spec_id":...}``.

    spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
    """
    svc = _get_service()
    result = svc.migrate(kb_id=kb_id, target_backend=target_backend)
    return json.dumps(result)


# ===================================================================
# Read-Path Tools (Always Available)
# ===================================================================


@mcp.tool()
@_with_tool_timeout
def kb_list() -> str:
    """List all knowledgebases.

    Returns a JSON array of knowledgebase objects.
    """
    svc = _get_service()
    kbs = svc.list_kbs()
    return _serialize_model_list(kbs)


@mcp.tool()
@_with_tool_timeout
def kb_info(kb_id: str) -> str:
    """Get knowledgebase details including source and page counts.

    Parameters:
        kb_id: ID of the knowledgebase.

    Returns a JSON object of the knowledgebase with counts.
    """
    svc = _get_service()
    kb = svc.get_kb(kb_id)
    if kb is None:
        raise ValueError(f"Knowledgebase '{kb_id}' not found")
    return kb.model_dump_json()


@mcp.tool()
@_with_tool_timeout
def kb_query(kb_id: str, text: str, top_k: int | None = None) -> str:
    """Semantic (vector similarity) query across a knowledgebase.

    Parameters:
        kb_id: ID of the knowledgebase to search.
        text: The query text.
        top_k: Maximum number of results to return; defaults to ``query_default_top_k`` if omitted.

    Returns a JSON array of search results. When the embedder backend is
    unreachable or exceeds its wall-clock bound, returns a single JSON
    object instead: ``{"error": "embed_timeout"|"embed_unreachable",
    "model": ..., "phase": "embed_query", "latency_ms": ..., "detail": ...}``.
    """
    svc = _get_service()
    try:
        results = svc.query(kb_id, text, top_k)
    except EmbedderUnavailableError as exc:
        return json.dumps(exc.to_payload())
    return _serialize_dataclass_list(results)


@mcp.tool()
@_with_tool_timeout
def kb_search(kb_id: str, text: str, top_k: int | None = None) -> str:
    """Keyword search across a knowledgebase.

    Parameters:
        kb_id: ID of the knowledgebase to search.
        text: The search text.
        top_k: Maximum number of results to return; defaults to ``query_default_top_k`` if omitted.

    Returns a JSON array of search results.
    """
    svc = _get_service()
    results = svc.search(kb_id, text, top_k)
    return _serialize_dataclass_list(results)


@mcp.tool()
@_with_tool_timeout
def kb_get_page(page_id: str) -> str:
    """Get a wiki page by ID.

    Parameters:
        page_id: ID of the page to retrieve.

    Returns a JSON object of the page.
    """
    svc = _get_service()
    page = svc.get_page(page_id)
    if page is None:
        raise ValueError(f"Page '{page_id}' not found")
    return page.model_dump_json()


@mcp.tool()
@_with_tool_timeout
def kb_list_pages(kb_id: str, page_type: str = "") -> str:
    """List wiki pages in a knowledgebase, optionally filtered by type.

    Parameters:
        kb_id: ID of the knowledgebase.
        page_type: Filter by page type (entity, concept, summary, index, comparison, synthesis). Empty string for all.

    Each page object includes the standard wiki page fields plus the following
    additive fields drawn from the first source in source_ids (first entry in
    list order, not chronological):

      - page_id:    alias of the page's id field (computed)
      - source_type: source kind (e.g. "file", "website") from source_ids[0]
      - uri:        source URI from source_ids[0]
      - dedup_key:  dedup key from source_ids[0]

    These fields are None when source_ids is empty, and are only populated by
    this endpoint — get_page, direct model construction, and direct DB reads
    leave them as None.

    Returns a JSON array of page objects.
    """
    svc = _get_service()
    pt: PageType | None = None
    if page_type:
        pt = PageType(page_type)
    pages = svc.list_pages(kb_id, pt)
    return _serialize_model_list(pages)


@mcp.tool()
@_with_tool_timeout
def kb_get_source(source_id: str) -> str:
    """Get source details by ID.

    Parameters:
        source_id: ID of the source to retrieve.

    Returns a JSON object of the source.
    """
    svc = _get_service()
    source = svc.get_source(source_id)
    if source is None:
        raise ValueError(f"Source '{source_id}' not found")
    return source.model_dump_json()


@mcp.tool()
@_with_tool_timeout
def kb_list_sources(kb_id: str) -> str:
    """List all sources in a knowledgebase.

    Parameters:
        kb_id: ID of the knowledgebase.

    Returns a JSON array of source objects.
    """
    svc = _get_service()
    sources = svc.list_sources(kb_id)
    return _serialize_model_list(sources)


@mcp.tool()
@_with_tool_timeout
def kb_get_links(page_id: str, direction: str = "outbound") -> str:
    """Get the link graph for a wiki page.

    Parameters:
        page_id: ID of the page.
        direction: Link direction — "outbound" (default) or "inbound".

    Returns a JSON array of linked page objects.
    """
    svc = _get_service()
    if direction not in ("outbound", "inbound"):
        raise ValueError(f"Invalid direction '{direction}'; must be 'outbound' or 'inbound'")
    pages = svc.get_links(page_id, direction)
    return _serialize_model_list(pages)


@mcp.tool()
@_with_tool_timeout
def kb_pipeline_status(kb_id: str) -> str:
    """Check the pipeline status for a knowledgebase.

    Parameters:
        kb_id: ID of the knowledgebase.

    Returns a JSON array of pipeline run objects.
    """
    svc = _get_service()
    runs = svc.get_pipeline_status(kb_id)
    return _serialize_model_list(runs)


# ===================================================================
# Config tools
# ===================================================================


def _resolved_via(env_name: str) -> str:
    return "env_override" if os.environ.get(env_name) else "default"


def _unflatten(flat: dict[str, object]) -> dict[str, object]:
    """Inverse of _flatten_dotted — restore a nested dict from dotted keys."""
    flat_to_dot = {v: k for k, v in DOT_TO_FLAT.items()}
    out: dict[str, object] = {}
    for flat_name, value in flat.items():
        dotted = flat_to_dot.get(flat_name, flat_name)
        segments = dotted.split(".")
        cursor = out
        for segment in segments[:-1]:
            cursor = cursor.setdefault(segment, {})  # type: ignore[assignment]
        cursor[segments[-1]] = value
    return out


def _provenance() -> dict[str, str]:
    """Determine which layer won for each configurable field.

    Returns a dict keyed by dotted path → one of
    ``"env"``, ``"project_json"``, ``"user_json"``, ``"default"``.
    """
    env_prefix = "AGENT_KB_"
    env_keys_present = {
        name[len(env_prefix):].lower()
        for name in os.environ
        if name.startswith(env_prefix)
    }
    user_dict = NestedJsonConfigSettingsSource(
        Settings, path=resolve_user_config_path()
    )()
    project_dict = NestedJsonConfigSettingsSource(
        Settings, path=resolve_project_config_path()
    )()

    result: dict[str, str] = {}
    for dotted, flat_name in DOT_TO_FLAT.items():
        if flat_name in env_keys_present:
            result[dotted] = "env"
        elif flat_name in project_dict:
            result[dotted] = "project_json"
        elif flat_name in user_dict:
            result[dotted] = "user_json"
        else:
            result[dotted] = "default"
    return result


@mcp.tool()
@_with_tool_timeout
def kb_config_path() -> str:
    """Return the on-disk paths used for user- and project-level config.

    Paths are returned whether or not the files exist. ``resolved_via`` is
    ``"env_override"`` when an ``AGENT_KB_*_CONFIG`` env var supplied the path,
    otherwise ``"default"``.
    """
    user = resolve_user_config_path()
    project = resolve_project_config_path()
    payload = {
        "user": {
            "path": str(user),
            "exists": user.exists(),
            "resolved_via": _resolved_via("AGENT_KB_USER_CONFIG"),
        },
        "project": {
            "path": str(project),
            "exists": project.exists(),
            "resolved_via": _resolved_via("AGENT_KB_PROJECT_CONFIG"),
        },
    }
    return json.dumps(payload)


@mcp.tool()
@_with_tool_timeout
def kb_config_show(scope: str = "merged") -> str:
    """Return configured values.

    Parameters:
        scope: One of "merged", "user", "project", "env", "defaults".

    ``merged`` also returns a ``provenance`` map naming which layer won for
    each key.
    """
    valid = {"merged", "user", "project", "env", "defaults"}
    if scope not in valid:
        raise ValueError(f"scope must be one of {sorted(valid)}, got {scope!r}")

    if scope == "merged":
        cfg = load_settings()
        values = {flat_name: getattr(cfg, flat_name) for flat_name in DOT_TO_FLAT.values()}
        return json.dumps({
            "values": _unflatten(values),
            "provenance": _provenance(),
        }, default=str)

    if scope == "user":
        return json.dumps(
            _unflatten(NestedJsonConfigSettingsSource(Settings, path=resolve_user_config_path())()),
            default=str,
        )
    if scope == "project":
        return json.dumps(
            _unflatten(NestedJsonConfigSettingsSource(Settings, path=resolve_project_config_path())()),
            default=str,
        )
    if scope == "env":
        from pydantic import TypeAdapter

        env_prefix = "AGENT_KB_"
        env_map: dict[str, object] = {}
        for flat_name in DOT_TO_FLAT.values():
            env_key = env_prefix + flat_name.upper()
            if env_key in os.environ:
                raw = os.environ[env_key]
                field = Settings.model_fields.get(flat_name)
                if field is not None and field.annotation is not None:
                    try:
                        env_map[flat_name] = TypeAdapter(field.annotation).validate_python(raw)
                    except Exception:
                        env_map[flat_name] = raw
                else:
                    env_map[flat_name] = raw
        return json.dumps(_unflatten(env_map), default=str)
    # scope == "defaults"
    defaults_flat: dict[str, object] = {}
    for flat_name in DOT_TO_FLAT.values():
        field = Settings.model_fields.get(flat_name)
        if field is None:
            continue
        if field.default_factory is not None:
            default = field.default_factory()
        else:
            default = field.default
        if default is PydanticUndefined:
            continue
        defaults_flat[flat_name] = default
    return json.dumps(_unflatten(defaults_flat), default=str)


@mcp.tool()
@_with_tool_timeout
def kb_config_get(key: str) -> str:
    """Return one setting by dotted path (e.g. ``embedding.model``).

    Response includes ``value``, ``provenance`` (which layer won), and
    ``effective_type`` — the runtime type name from
    ``type(value).__name__``. For unset optional fields (``export_path``,
    ``pinecone.index``, etc.) this is ``"NoneType"``, which lets callers
    distinguish a missing value from an empty string or zero.
    """
    if key not in DOT_TO_FLAT:
        valid_list = ", ".join(sorted(DOT_TO_FLAT))
        raise ValueError(
            f"unknown_key: '{key}' is not a valid config key. Valid keys: {valid_list}"
        )

    cfg = load_settings()
    flat = DOT_TO_FLAT[key]
    value = getattr(cfg, flat)
    payload = {
        "key": key,
        "value": value,
        "provenance": _provenance().get(key, "default"),
        "effective_type": type(value).__name__,
    }
    return json.dumps(payload, default=str)


@mcp.tool()
@_with_tool_timeout
def kb_config_set(scope: str, key: str, value: object) -> str:
    """Write a single setting to the user- or project-level config file.

    Parameters:
        scope: ``"user"`` or ``"project"``.
        key: Dotted path, e.g. ``"embedding.model"``.
        value: New value (type validated by pydantic on the dry-run).

    Behavior: writes ``<file>.tmp`` beside the target, runs validation on
    the candidate, then atomically replaces the real file on success.  On
    validation failure, the real file is untouched and the tmp is deleted.
    """
    if scope not in {"user", "project"}:
        raise ValueError(f"scope must be 'user' or 'project', got {scope!r}")
    if key not in DOT_TO_FLAT:
        valid = ", ".join(sorted(DOT_TO_FLAT))
        raise ValueError(f"unknown config key '{key}'. Valid: {valid}")
    for segment in key.split("."):
        if segment in FORBIDDEN_KEYS:
            raise ValueError(
                f"key '{key}' contains forbidden segment '{segment}' — "
                f"secrets must be set via env, never written to disk."
            )

    target = resolve_user_config_path() if scope == "user" else resolve_project_config_path()
    write_candidate_and_validate(target, key, value, scope=scope)

    # Refresh the service/settings after the successful write so the next
    # tool call sees the new value.
    #
    # NOTE: This invalidates only the server.py module-level `_service`
    # cache.  Any `Settings` object cached in another module (e.g., held
    # by a test fixture or a caller outside server.py) is NOT refreshed.
    # The existing callers all go through `_get_service()` or
    # `load_settings()`, which both re-read on demand, so this is safe.
    global _service
    _service = None

    return kb_config_get(key)


@mcp.tool()
@_with_tool_timeout
def kb_config_validate() -> str:
    """Dry-run config resolution and report per-file status.

    Does not mutate server state.  Returns ``status`` = ``"ok"``,
    ``"missing"``, or ``"error"`` for each file.  When both are
    ``"ok"``/``"missing"``, also returns the merged values + provenance.
    """
    def _check(path: Path) -> dict[str, object]:
        if not path.exists():
            return {"status": "missing", "path": str(path)}
        try:
            NestedJsonConfigSettingsSource(Settings, path=path)()
        except Exception as e:  # noqa: BLE001 — we want the full error text
            return {"status": "error", "path": str(path), "error": str(e)}
        return {"status": "ok", "path": str(path)}

    user_path = resolve_user_config_path()
    project_path = resolve_project_config_path()
    user_status = _check(user_path)
    project_status = _check(project_path)

    payload: dict[str, object] = {
        "user": user_status,
        "project": project_status,
    }

    if user_status["status"] != "error" and project_status["status"] != "error":
        try:
            cfg = load_settings()
            values = {flat: getattr(cfg, flat) for flat in DOT_TO_FLAT.values()}
            payload["merged"] = {
                "values": _unflatten(values),
                "provenance": _provenance(),
            }
        except Exception as e:  # noqa: BLE001
            payload["merged_error"] = str(e)

    return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server.

    Performs a startup precheck of the configuration so that missing or
    misconfigured paths fail immediately with a structured single-line
    JSON message on stderr, rather than surfacing on the first
    ``kb_*`` MCP call and confusing the caller. ``sys.exit(1)`` on
    failure ensures the launcher / supervisor sees a non-zero exit
    code.
    """
    try:
        Settings().resolve_paths()
    except (ValidationError, FileNotFoundError, NotADirectoryError) as exc:
        payload = _classify_config_error(exc)
        sys.stderr.write(json.dumps(payload) + "\n")
        sys.stderr.flush()
        sys.exit(1)
    mcp.run()


if __name__ == "__main__":
    main()
