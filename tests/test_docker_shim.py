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


def test_execs_into_container_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_server.shutil, "which", lambda _c: "/usr/bin/docker")

    def fake_inspect(argv, **_kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = '"running"\n'
        return result

    monkeypatch.setattr(subprocess, "run", fake_inspect)

    captured_argv: list[str] = []

    def fake_execv(path: str, args: list[str]) -> None:
        captured_argv[:] = [path, *args[1:]]
        # Don't actually exec — raise to break out of main()
        raise RuntimeError("execv called")

    monkeypatch.setattr(run_server.os, "execvp", fake_execv)
    with pytest.raises(RuntimeError, match="execv called"):
        run_server.main()
    assert captured_argv[0] == "/usr/bin/docker"
    assert "exec" in captured_argv
    assert "-i" in captured_argv
    assert "agent-knowledgebase" in captured_argv
    assert "agent_knowledgebase.server" in " ".join(captured_argv)
