"""Shared helpers for the integration smoke + diagnostic scripts.

These scripts run against a real MCP server / real embedder / real persistent
KB, so they need to resolve user-specific values (saves dir, KB id, YT cache
DB path) at runtime rather than hardcoding them. All resolution funnels
through this module so adding new env vars or fallback strategies is a
one-line change.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def plugin_root() -> Path:
    """Return the plugin root, computed from this file's location.

    `tests/integration/_helpers.py` → `parents[2]` is the plugin root.
    """
    return Path(__file__).resolve().parents[2]


def ensure_src_on_path() -> None:
    """Insert `<plugin_root>/src` at the front of `sys.path` if missing.

    The smoke scripts that bypass MCP and call the service in-process need
    `agent_knowledgebase` importable without an editable install.
    """
    src = plugin_root() / "src"
    p = str(src)
    if p not in sys.path:
        sys.path.insert(0, p)


def require_env(name: str, hint: str | None = None) -> str:
    """Return the env var or exit 2 with a copy-pasteable hint."""
    val = os.environ.get(name)
    if val:
        return val
    sys.stderr.write(f"\nERROR: {name} is not set.\n")
    if hint:
        sys.stderr.write(f"  Hint: {hint}\n")
    sys.exit(2)


def resolve_kb_id() -> str:
    """Resolve the KB id this run should target.

    Priority:
      1. `AGENT_KB_TEST_KB_ID` env var (explicit override).
      2. Look up `AGENT_KB_TEST_KB_NAME` (default `claude-rag`) via the
         in-process `KnowledgebaseService` and use that KB's id.

    Falling back to a name lookup keeps the scripts portable across machines
    while letting an operator pin a specific UUID when they want one.
    """
    explicit = os.environ.get("AGENT_KB_TEST_KB_ID")
    if explicit:
        return explicit

    name = os.environ.get("AGENT_KB_TEST_KB_NAME", "claude-rag")

    ensure_src_on_path()
    require_env(
        "AGENT_KB_SAVES_DIR",
        "set it to your persistent KB saves directory before running.",
    )
    from agent_knowledgebase.config import Settings  # noqa: E402
    from agent_knowledgebase.services.knowledgebase import (  # noqa: E402
        KnowledgebaseService,
    )

    svc = KnowledgebaseService(Settings())
    for kb in svc.list_kbs():
        if kb.name == name:
            return kb.id

    sys.stderr.write(
        f"\nERROR: no KB named {name!r} found in AGENT_KB_SAVES_DIR.\n"
        f"  Run smoke_01_create_kb.py first, or set AGENT_KB_TEST_KB_ID.\n"
    )
    sys.exit(2)


def resolve_yt_db() -> Path:
    """Resolve the YouTube cache sqlite path.

    Required by `smoke_02_export_transcripts.py` only. The expected schema is
    the one produced by the `youtube-mcp` plugin's local cache (tables
    `videos` and `transcripts`, keyed by `video_id`).
    """
    p = require_env(
        "AGENT_KB_TEST_YT_DB",
        "set it to a youtube-mcp cache sqlite (e.g. .../youtube-data.db).",
    )
    path = Path(p)
    if not path.exists():
        sys.stderr.write(f"\nERROR: AGENT_KB_TEST_YT_DB does not exist: {path}\n")
        sys.exit(2)
    return path


def transcripts_dir() -> Path:
    """Return the bundled transcript-fixture directory."""
    return Path(__file__).resolve().parent / "fixtures" / "transcripts"


def transcripts_manifest_path() -> Path:
    """Return the bundled transcript manifest path."""
    return Path(__file__).resolve().parent / "fixtures" / "transcripts_manifest.json"
