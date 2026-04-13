"""User- and project-level JSON config file support for agent-knowledgebase.

This module is independent of :mod:`agent_knowledgebase.config` to avoid
circular imports — :class:`Settings` imports from here, not vice-versa.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("agent_knowledgebase.config")

# Module-level overrides consulted by resolve_*_config_path().  Used by
# load_settings() to perform a dry-run against a candidate file without
# mutating the real on-disk state.  Public API is the helper functions —
# tests should prefer those.
_USER_CONFIG_PATH_OVERRIDE: Path | None = None
_PROJECT_CONFIG_PATH_OVERRIDE: Path | None = None


def resolve_user_config_path() -> Path:
    """Return the absolute path to the user-level config file.

    Discovery order:
      1. Module-level override set by :func:`load_settings` (test + dry-run aid).
      2. ``$AGENT_KB_USER_CONFIG`` environment variable.
      3. ``~/.agent-kb/config.json`` default.
    """
    if _USER_CONFIG_PATH_OVERRIDE is not None:
        return _USER_CONFIG_PATH_OVERRIDE
    env_value = os.environ.get("AGENT_KB_USER_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / ".agent-kb" / "config.json"


def resolve_project_config_path() -> Path:
    """Return the absolute path to the project-level config file.

    Discovery order:
      1. Module-level override set by :func:`load_settings`.
      2. ``$AGENT_KB_PROJECT_CONFIG`` explicit file path.
      3. ``<project_root>/.agent-kb/config.json`` where <project_root> is
         derived from, in order: ``$AGENT_KB_PROJECT_DIR``,
         ``$CLAUDE_PROJECT_DIR``, ``$PWD``, then :func:`os.getcwd`.
    """
    if _PROJECT_CONFIG_PATH_OVERRIDE is not None:
        return _PROJECT_CONFIG_PATH_OVERRIDE
    env_file = os.environ.get("AGENT_KB_PROJECT_CONFIG")
    if env_file:
        return Path(env_file).expanduser()

    for env_name in ("AGENT_KB_PROJECT_DIR", "CLAUDE_PROJECT_DIR", "PWD"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser() / ".agent-kb" / "config.json"

    return Path(os.getcwd()) / ".agent-kb" / "config.json"
