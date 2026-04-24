#!/usr/bin/env python3
"""
agent-knowledgebase Pattern A launcher (SCAFFOLD STUB).

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        architecture.components[0] "bin/run_server.py — Pattern A single Python launcher"

This file is the entry-point referenced by .mcp.json once Phase 1 ships:

    {
      "mcpServers": {
        "agent-knowledgebase": {
          "command": "python",
          "args": ["${CLAUDE_PLUGIN_ROOT}/bin/run_server.py"]
        }
      }
    }

Cross-platform: invoked as `python <this_file>`, so PATHEXT and shebang
issues are sidestepped — the `python` on PATH does its own arg expansion.

PHASE 1 SCOPE (this scaffold ships only the python_version check + execv
hand-off; the venv-bootstrap implementation is Phase 1 work):
  - locate or create ${CLAUDE_PLUGIN_ROOT}/.venv via stdlib `venv` module
  - acquire ${CLAUDE_PLUGIN_ROOT}/.venv.lock via filelock to make first-run
    pip install crash-safe under concurrent worker spawns
  - SHA256(requirements.lock) sentinel reuse — if .venv/.req-sha matches,
    skip pip install
  - 30s pip-install network timeout
  - three structured stderr error modes: python_version, network_unreachable,
    read_only_filesystem
  - AGENT_KB_VENDORED_DEPS=/path/to/wheels offline escape-hatch
  - os.execv venv-python -m agent_knowledgebase.server

DO NOT add a bash/cmd companion. Single .py file is the design.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Spec_id pinned for traceability — future agents grep this comment to find
# the launcher that implements the v2.1 Pattern A design.
_SPEC_ID = "70ab2170-381a-4657-bcd1-28a40c6f369b"
_SPEC_VERSION = "2.1"


def _emit_stderr_error(error_token: str, detail: str, **extra: object) -> None:
    """
    Emit a structured single-line JSON error to stderr.

    The three canonical error_token values per spec are:
      - "python_version"      Python<3.11 detected
      - "network_unreachable" PyPI 30s timeout — set AGENT_KB_VENDORED_DEPS
      - "read_only_filesystem" CLAUDE_PLUGIN_ROOT not writable

    Phase 1 will populate the latter two; this scaffold only emits
    python_version.
    """
    payload: dict[str, object] = {
        "error": error_token,
        "detail": detail,
        "launcher": "agent-knowledgebase/bin/run_server.py",
        "spec_id": _SPEC_ID,
        "spec_version": _SPEC_VERSION,
    }
    payload.update(extra)
    sys.stderr.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stderr.flush()


def _resolve_plugin_root() -> Path:
    """
    Compute CLAUDE_PLUGIN_ROOT.

    Prefer the env var set by Claude Code at MCP launch; fall back to
    `<this_file>.parent.parent` so the launcher works in dev when invoked
    directly via `python bin/run_server.py`.
    """
    env_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent


def _resolve_venv_python(plugin_root: Path) -> Path:
    """
    Compute the path to the venv's python interpreter.

    Windows: <plugin_root>/.venv/Scripts/python.exe
    Unix:    <plugin_root>/.venv/bin/python
    """
    venv_dir = plugin_root / ".venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _check_python_version() -> None:
    """
    Fail-fast if Python<3.11. Required floor per spec assumptions.
    """
    if sys.version_info < (3, 11):
        _emit_stderr_error(
            "python_version",
            f"Python>=3.11 required; detected {sys.version_info.major}.{sys.version_info.minor}",
            detected=f"{sys.version_info.major}.{sys.version_info.minor}",
            required=">=3.11",
        )
        sys.exit(1)


def main() -> None:
    """
    Pattern A launcher entry point.

    Phase 1 IMPLEMENTATION TODO list (this scaffold does NOT implement these):

        TODO[Phase 1]: Acquire <plugin_root>/.venv.lock via stdlib filelock
                       (fcntl on Unix, msvcrt on Windows) so concurrent
                       Claude Code worker spawns don't race the venv create.

        TODO[Phase 1]: Cygwin/Git-Bash path normalization
                       — sys.platform=='win32' AND env('MSYSTEM') set means
                       CLAUDE_PLUGIN_ROOT may contain forward slashes from
                       Git-Bash. Normalize via Path(...).resolve() but verify
                       the venv-python path uses backslashes for subprocess.

        TODO[Phase 1]: Compute SHA256(requirements.lock) and compare against
                       <plugin_root>/.venv/.req-sha sentinel. Skip pip install
                       on match.

        TODO[Phase 1]: If sentinel mismatch, run `pip install -r
                       requirements.lock` (or AGENT_KB_VENDORED_DEPS path
                       via --no-index --find-links) inside venv with 30s
                       network timeout. On timeout: emit structured stderr
                       {"error":"network_unreachable", ...} and exit 1.

        TODO[Phase 1]: Verify CLAUDE_PLUGIN_ROOT is writable via
                       os.access(plugin_root, os.W_OK). On failure emit
                       {"error":"read_only_filesystem", ...} and exit 1.

        TODO[Phase 1]: After successful venv setup, write new sentinel and
                       fall through to the os.execv hand-off below.

        TODO[Phase 1]: AGENT_KB_VENDORED_DEPS handling — if env var set, use
                       `pip install --no-index --find-links=$AGENT_KB_VENDORED_DEPS`
                       so corporate / air-gapped installs work.
    """
    _check_python_version()

    plugin_root = _resolve_plugin_root()
    venv_python = _resolve_venv_python(plugin_root)

    # In this scaffold the venv may not yet exist; Phase 1 will guarantee it.
    # Until then, fall back to the running interpreter so the launcher is
    # at least directly runnable for development.
    if not venv_python.exists():
        # TODO[Phase 1]: bootstrap the venv here per the TODOs above.
        # For now, fall back to sys.executable so dev runs work.
        venv_python = Path(sys.executable)

    # Hand off to the server module. os.execv replaces this process so the
    # launcher does not stay resident.
    args = [str(venv_python), "-m", "agent_knowledgebase.server"]
    os.execv(args[0], args)


if __name__ == "__main__":
    main()
