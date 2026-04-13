"""MCP server entry point for agent-knowledgebase."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

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
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService

mcp = FastMCP(
    "agent-knowledgebase",
    instructions="Personal knowledgebase plugin — ingest sources, build wiki artifacts, query with vector search",
)

# ---------------------------------------------------------------------------
# Lazy initialization of the service singleton
# ---------------------------------------------------------------------------

_service: KnowledgebaseService | None = None


def _get_service() -> KnowledgebaseService:
    global _service
    if _service is None:
        config = Settings().resolve_paths()
        _service = KnowledgebaseService(config)
    return _service


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


# ===================================================================
# Write-Path Tools (Pipeline-Gated)
# ===================================================================


@mcp.tool()
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
def kb_ingest(kb_id: str, source_type: str, uri: str, metadata: str = "{}") -> str:
    """Ingest a source into a knowledgebase.

    Parameters:
        kb_id: ID of the target knowledgebase.
        source_type: Type of source (file, directory, codebase, website, sql_database, git_history, api_endpoint).
        uri: Location or path of the source.
        metadata: JSON string of additional metadata.

    Returns a JSON object of the created source.
    """
    svc = _get_service()
    st = SourceType(source_type)
    meta = json.loads(metadata)
    source = svc.ingest_source(kb_id, st, uri, meta)
    return source.model_dump_json()


@mcp.tool()
def kb_ingest_batch(kb_id: str, sources: str) -> str:
    """Batch ingest multiple sources into a knowledgebase.

    Parameters:
        kb_id: ID of the target knowledgebase.
        sources: JSON array of objects, each with keys: source_type, uri, metadata (optional).

    Returns a JSON array of created source objects.
    """
    svc = _get_service()
    source_defs = json.loads(sources)
    results = []
    for item in source_defs:
        st = SourceType(item["source_type"])
        uri = item["uri"]
        meta = item.get("metadata", {})
        source = svc.ingest_source(kb_id, st, uri, meta)
        results.append(source)
    return _serialize_model_list(results)


@mcp.tool()
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


# ===================================================================
# Read-Path Tools (Always Available)
# ===================================================================


@mcp.tool()
def kb_list() -> str:
    """List all knowledgebases.

    Returns a JSON array of knowledgebase objects.
    """
    svc = _get_service()
    kbs = svc.list_kbs()
    return _serialize_model_list(kbs)


@mcp.tool()
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
def kb_query(kb_id: str, text: str, top_k: int = 10) -> str:
    """Semantic (vector similarity) query across a knowledgebase.

    Parameters:
        kb_id: ID of the knowledgebase to search.
        text: The query text.
        top_k: Maximum number of results to return.

    Returns a JSON array of search results.
    """
    svc = _get_service()
    results = svc.query(kb_id, text, top_k)
    return _serialize_dataclass_list(results)


@mcp.tool()
def kb_search(kb_id: str, text: str, top_k: int = 10) -> str:
    """Keyword search across a knowledgebase.

    Parameters:
        kb_id: ID of the knowledgebase to search.
        text: The search text.
        top_k: Maximum number of results to return.

    Returns a JSON array of search results.
    """
    svc = _get_service()
    results = svc.search(kb_id, text, top_k)
    return _serialize_dataclass_list(results)


@mcp.tool()
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
def kb_list_pages(kb_id: str, page_type: str = "") -> str:
    """List wiki pages in a knowledgebase, optionally filtered by type.

    Parameters:
        kb_id: ID of the knowledgebase.
        page_type: Filter by page type (entity, concept, summary, index, comparison, synthesis). Empty string for all.

    Returns a JSON array of page objects.
    """
    svc = _get_service()
    pt: PageType | None = None
    if page_type:
        pt = PageType(page_type)
    pages = svc.list_pages(kb_id, pt)
    return _serialize_model_list(pages)


@mcp.tool()
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
    global _service
    _service = None

    return kb_config_get(key)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
