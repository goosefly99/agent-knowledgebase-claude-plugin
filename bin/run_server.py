#!/usr/bin/env python3
"""agent-knowledgebase Docker shim launcher (v0.13.0+).

Replaces the v0.8.0 stdlib-venv bootstrap. The plugin now requires a
running Docker container named ``agent-knowledgebase`` (started via
``cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose up -d`` so any
host-specific ``docker-compose.override.yml`` is auto-loaded). This shim:

  1. Verifies ``docker`` is on PATH.
  2. Verifies the ``agent-knowledgebase`` container is in state ``running``.
  3. ``os.execvp("docker", ["docker", "exec", "-i", "agent-knowledgebase",
     "python", "-m", "agent_knowledgebase.server"])`` pipes the MCP stdio
     into a fresh server process inside the container.

If any step fails, the shim emits a single-line JSON document on stderr
and exits 1 — Claude Code's MCP transport surfaces the structured payload
to the user. Canonical error tokens:

  docker_not_installed     ``docker`` is not on PATH
  container_not_running    container missing or not in ``running`` state
  inspect_failed           docker inspect raised an unexpected error
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

CONTAINER_NAME = "agent-knowledgebase"
COMPOSE_HINT = (
    "cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose up -d"
)


def _emit_stderr_error(error_token: str, detail: str, **extra: object) -> None:
    payload: dict[str, object] = {
        "error": error_token,
        "detail": detail,
        "launcher": "agent-knowledgebase/bin/run_server.py",
    }
    payload.update(extra)
    sys.stderr.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stderr.flush()


def is_container_running(name: str) -> bool:
    """Return True iff the named container exists and is in state 'running'."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{json .State.Status}}", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    if result.returncode != 0:
        return False
    try:
        status = json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return False
    return status == "running"


def main() -> None:
    docker_path = shutil.which("docker")
    if docker_path is None:
        _emit_stderr_error(
            "docker_not_installed",
            "Docker CLI not found on PATH. Install Docker Desktop "
            "(or the docker engine on Linux) and re-run.",
        )
        sys.exit(1)

    if not is_container_running(CONTAINER_NAME):
        _emit_stderr_error(
            "container_not_running",
            f"The agent-knowledgebase container is not running. Start it with: "
            f"{COMPOSE_HINT}",
            container=CONTAINER_NAME,
            start_command=COMPOSE_HINT,
        )
        sys.exit(1)

    args = [
        "docker",
        "exec",
        "-i",
        CONTAINER_NAME,
        "python",
        "-m",
        "agent_knowledgebase.server",
    ]
    os.execvp(docker_path, args)


if __name__ == "__main__":
    main()
