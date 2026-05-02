"""bin/run_server.py: structured error emission tests.

The shim must emit one structured single-line JSON error to stderr
for each failure mode (docker not installed, container not running,
exec failed) and exit 1 — never propagate a raw exception or hang.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
sys.path.insert(0, str(_BIN_DIR))
import run_server  # type: ignore[import-not-found]  # noqa: E402


def _capture_stderr_json(capsys: pytest.CaptureFixture[str]) -> dict:
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    return json.loads(line)


def test_emits_docker_not_installed_when_docker_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_which(_cmd: str) -> None:
        return None

    monkeypatch.setattr(run_server.shutil, "which", fake_which)
    with pytest.raises(SystemExit) as exc:
        run_server.main()
    assert exc.value.code == 1
    payload = _capture_stderr_json(capsys)
    assert payload["error"] == "docker_not_installed"


def test_emits_container_not_running_when_inspect_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_server.shutil, "which", lambda _c: "/usr/bin/docker")

    def fake_run(argv, **_kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "Error: No such container: agent-knowledgebase\n"
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        run_server.main()
    assert exc.value.code == 1
    payload = _capture_stderr_json(capsys)
    assert payload["error"] == "container_not_running"
    assert "docker compose" in payload["detail"]


def test_emits_container_not_running_when_status_is_exited(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_server.shutil, "which", lambda _c: "/usr/bin/docker")

    def fake_run(argv, **_kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = '"exited"\n'
        result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        run_server.main()
    assert exc.value.code == 1
    payload = _capture_stderr_json(capsys)
    assert payload["error"] == "container_not_running"


def test_proxies_into_container_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shim must proxy via subprocess.run (not os.execvp) so the launcher
    PID stays alive throughout the MCP session.

    Background: on Windows ``os.execvp`` is _P_OVERLAY which kills the
    original PID and spawns docker.exe under a new PID. Claude Code's
    MCP transport tracks the original PID; when it dies the harness
    closes the stdio pipes and the MCP session is torn down before
    ``tools/list`` ever runs.
    """
    monkeypatch.setattr(run_server.shutil, "which", lambda _c: "/usr/bin/docker")

    captured_argvs: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        captured_argvs.append(list(argv))
        result = MagicMock()
        result.returncode = 0
        # First call is `docker inspect` for is_container_running()
        result.stdout = '"running"\n'
        return result

    monkeypatch.setattr(run_server.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        run_server.main()
    assert exc.value.code == 0

    # subprocess.run was invoked twice: once to inspect, once to exec.
    assert len(captured_argvs) == 2
    inspect_argv, exec_argv = captured_argvs

    assert "inspect" in inspect_argv

    assert exec_argv[0] == "/usr/bin/docker"
    assert "exec" in exec_argv
    assert "-i" in exec_argv
    assert "agent-knowledgebase" in exec_argv
    assert "agent_knowledgebase.server" in " ".join(exec_argv)
