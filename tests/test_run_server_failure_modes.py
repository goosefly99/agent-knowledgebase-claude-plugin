"""Tests for the structured stderr error modes of the Pattern A launcher.

Spec: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
      implementation.phases[1] "Phase 1 — Pattern A single-launcher runtime
      bootstrap" — deliverable ``tests/test_run_server_failure_modes.py``.

Canonical structured-stderr error modes surfaced by the launcher:

    {"error":"python_version",         "detected":"3.10", "required":">=3.11", ...}
    {"error":"network_unreachable",    "timeout_seconds":30, ...}
    {"error":"read_only_filesystem",   "plugin_root":"...", ...}
    {"error":"bootstrap_lock_timeout", "timeout_seconds":60, "lock_path":"...", ...}
    {"error":"venv_create_failed",     "venv_dir":"...", "exception_type":"OSError", ...}
    {"error":"pip_install_failed",     "returncode":1, "stderr":"...", ...}
    {"error":"vendored_deps_invalid",  "path":"...", ...}

Each emission must be a single-line JSON document carrying the spec_id
for traceability, and must end the process with ``SystemExit(1)``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LAUNCHER_PATH = _PROJECT_ROOT / "bin" / "run_server.py"


def _load_launcher_module():
    spec = importlib.util.spec_from_file_location(
        "run_server", _LAUNCHER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def launcher():
    return _load_launcher_module()


def _stderr_payload_lines(captured: str) -> list[dict]:
    """Parse stderr lines emitted by ``_emit_stderr_error`` into JSON dicts."""
    return [
        json.loads(line)
        for line in captured.strip().splitlines()
        if line.strip()
    ]


# -----------------------------------------------------------------------------
# Test 1 — Python<3.11 detected.
# -----------------------------------------------------------------------------


def test_python_version_under_3_11_emits_structured_stderr(
    launcher, capsys: pytest.CaptureFixture[str]
) -> None:
    """Patch ``sys.version_info`` to 3.10.x; assert structured stderr + exit.

    ``sys.version_info`` is a CPython-built type that cannot be
    instantiated from Python — calling ``sys.version_info.__class__(...)``
    raises ``TypeError: cannot create 'sys.version_info' instances``. The
    launcher only consults tuple comparison and ``.major`` / ``.minor``
    attribute access, both of which a stdlib ``namedtuple`` supports
    transparently. So we substitute a duck-typed namedtuple here.
    """
    from collections import namedtuple

    FakeVersion = namedtuple(
        "FakeVersion", ["major", "minor", "micro", "releaselevel", "serial"]
    )
    fake_version = FakeVersion(
        major=3, minor=10, micro=12, releaselevel="final", serial=0
    )
    with patch.object(launcher.sys, "version_info", fake_version):
        with pytest.raises(SystemExit) as excinfo:
            launcher._check_python_version()

    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    payloads = _stderr_payload_lines(captured.err)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["error"] == "python_version"
    assert payload["detected"] == "3.10"
    assert payload["required"] == ">=3.11"
    assert payload["spec_id"] == launcher._SPEC_ID
    assert payload["spec_version"] == launcher._SPEC_VERSION
    assert payload["launcher"] == "agent-knowledgebase/bin/run_server.py"


# -----------------------------------------------------------------------------
# Test 2 — pip install times out -> network_unreachable.
# -----------------------------------------------------------------------------


def test_pip_timeout_emits_network_unreachable(
    launcher, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Patch ``subprocess.run`` to raise ``TimeoutExpired``; assert payload."""
    venv_python = tmp_path / "venv-python"
    requirements = tmp_path / "requirements.lock"
    requirements.write_text("mcp==1.0.0\n")

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["pip", "install"], timeout=30)

    with patch.object(launcher.subprocess, "run", side_effect=raise_timeout):
        with pytest.raises(SystemExit) as excinfo:
            launcher._run_pip_install(venv_python, requirements)

    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    payloads = _stderr_payload_lines(captured.err)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["error"] == "network_unreachable"
    assert payload["timeout_seconds"] == launcher._PIP_INSTALL_TIMEOUT_SECONDS
    assert "AGENT_KB_VENDORED_DEPS" in payload["detail"]
    assert payload["spec_id"] == launcher._SPEC_ID


# -----------------------------------------------------------------------------
# Test 3 — read-only filesystem under CLAUDE_PLUGIN_ROOT.
# -----------------------------------------------------------------------------


def test_read_only_plugin_root_emits_structured_stderr(
    launcher, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Patch ``os.access`` to refuse write; assert structured stderr + exit."""
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()  # exists, so we don't trip the missing-dir branch

    def refuse_write(_path, _mode):
        return False

    with patch.object(launcher.os, "access", side_effect=refuse_write):
        with pytest.raises(SystemExit) as excinfo:
            launcher._check_writable(plugin_root)

    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    payloads = _stderr_payload_lines(captured.err)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["error"] == "read_only_filesystem"
    assert payload["plugin_root"] == str(plugin_root)
    assert "writable" in payload["detail"].lower()
    assert payload["spec_id"] == launcher._SPEC_ID


# -----------------------------------------------------------------------------
# Test 4 — missing CLAUDE_PLUGIN_ROOT emits the same token (sentinel for the
# user — they need to look at the same recovery hint).
# -----------------------------------------------------------------------------


def test_missing_plugin_root_emits_read_only_filesystem(
    launcher, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-existent plugin root surfaces the same ``read_only_filesystem``
    token because the user-visible recovery is identical (mount it
    writable / point it elsewhere).
    """
    missing = tmp_path / "not_a_real_plugin_dir"
    assert not missing.exists()

    with pytest.raises(SystemExit) as excinfo:
        launcher._check_writable(missing)
    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    payloads = _stderr_payload_lines(captured.err)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["error"] == "read_only_filesystem"
    assert payload["plugin_root"] == str(missing)


# -----------------------------------------------------------------------------
# Test 5 — Python>=3.11 passes the version check (regression guard).
# -----------------------------------------------------------------------------


def test_python_311_passes_version_check(launcher) -> None:
    """Sanity check: the host Python (>=3.11 per pyproject.toml) does not
    trip the ``python_version`` exit branch.
    """
    # Real sys.version_info, no patch.
    assert sys.version_info >= (3, 11)
    # Should not raise.
    launcher._check_python_version()


# -----------------------------------------------------------------------------
# Test 6 — bootstrap lock contention deadline -> bootstrap_lock_timeout.
# -----------------------------------------------------------------------------


def test_bootstrap_lock_timeout_emits_structured_stderr(
    launcher,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch the platform locking primitive to always refuse acquisition;
    assert the launcher emits ``bootstrap_lock_timeout`` and exits 1.

    We collapse the deadline + poll interval to near-zero so the test
    finishes in milliseconds rather than the production 60s budget.
    Both the Unix (``fcntl.flock``) and Windows (``msvcrt.locking``)
    branches must produce IDENTICAL structured stderr — that's the
    cross-platform contention semantics fix this test guards.
    """
    monkeypatch.setattr(launcher, "_BOOTSTRAP_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(launcher, "_BOOTSTRAP_LOCK_POLL_SECONDS", 0.01)

    lock_path = tmp_path / ".venv.lock"
    fh = open(lock_path, "a+")
    try:
        if sys.platform == "win32":
            def always_blocked(_fd, _mode, _nbytes):
                raise OSError("simulated contention")

            monkeypatch.setattr(launcher.msvcrt, "locking", always_blocked)
        else:
            def always_blocked(_fd, _flags):
                raise BlockingIOError("simulated contention")

            monkeypatch.setattr(launcher.fcntl, "flock", always_blocked)

        with pytest.raises(SystemExit) as excinfo:
            launcher._acquire_bootstrap_lock_or_exit(fh, lock_path)
    finally:
        fh.close()

    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    payloads = _stderr_payload_lines(captured.err)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["error"] == "bootstrap_lock_timeout"
    assert payload["timeout_seconds"] == 0.05
    assert payload["lock_path"] == str(lock_path)
    assert payload["spec_id"] == launcher._SPEC_ID
    assert "another process holds" in payload["detail"]


# -----------------------------------------------------------------------------
# Test 7 — venv.create failure -> venv_create_failed.
# -----------------------------------------------------------------------------


def test_venv_create_failure_emits_structured_stderr(
    launcher, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Patch ``venv.create`` to raise ``OSError("disk full")``; assert the
    launcher emits the ``venv_create_failed`` token instead of letting
    the raw traceback propagate.
    """
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()

    venv_python = (
        (plugin_root / ".venv" / "Scripts" / "python.exe")
        if sys.platform == "win32"
        else (plugin_root / ".venv" / "bin" / "python")
    )
    # Sanity: venv-python does NOT exist, so _ensure_venv will try to
    # call venv.create.
    assert not venv_python.exists()

    def raise_disk_full(*_args, **_kwargs):
        raise OSError("disk full")

    with patch.object(launcher.venv, "create", side_effect=raise_disk_full):
        with pytest.raises(SystemExit) as excinfo:
            launcher._ensure_venv(plugin_root, venv_python)

    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    payloads = _stderr_payload_lines(captured.err)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["error"] == "venv_create_failed"
    assert payload["venv_dir"] == str(plugin_root / ".venv")
    assert payload["exception_type"] == "OSError"
    assert "disk full" in payload["detail"]
    assert payload["spec_id"] == launcher._SPEC_ID


# -----------------------------------------------------------------------------
# Test 8 — pip install non-zero exit -> pip_install_failed.
# -----------------------------------------------------------------------------


def test_pip_install_nonzero_exit_emits_structured_stderr(
    launcher, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Patch ``subprocess.run`` to raise ``CalledProcessError(returncode=1)``
    with a captured stderr payload; assert the launcher emits the
    ``pip_install_failed`` token carrying the returncode and (truncated)
    stderr capture.
    """
    venv_python = tmp_path / "venv-python"
    requirements = tmp_path / "requirements.lock"
    requirements.write_text("mcp==1.0.0\n")

    # Build a stderr payload longer than the truncation budget so we can
    # also assert the truncation logic works.
    long_stderr = "ERROR: dependency conflict — " + ("X" * 600)

    def raise_called_process_error(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["pip", "install", "-r", "requirements.lock"],
            output="",
            stderr=long_stderr,
        )

    with patch.object(
        launcher.subprocess, "run", side_effect=raise_called_process_error
    ):
        with pytest.raises(SystemExit) as excinfo:
            launcher._run_pip_install(venv_python, requirements)

    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    payloads = _stderr_payload_lines(captured.err)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["error"] == "pip_install_failed"
    assert payload["returncode"] == 1
    # Truncated to 500 chars + the "...[truncated]" sentinel.
    assert payload["stderr"].endswith("...[truncated]")
    assert len(payload["stderr"]) <= 500 + len("...[truncated]")
    assert payload["spec_id"] == launcher._SPEC_ID


# -----------------------------------------------------------------------------
# Test 9 — AGENT_KB_VENDORED_DEPS path must exist -> vendored_deps_invalid.
# -----------------------------------------------------------------------------


def test_vendored_deps_path_must_exist(
    launcher,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set ``AGENT_KB_VENDORED_DEPS`` to a non-existent path; assert the
    launcher emits ``vendored_deps_invalid`` and exits 1 BEFORE pip
    is invoked. This is the up-front validation that saves users from
    pip's confusing "no matching distribution" wall-of-errors.
    """
    bogus = tmp_path / "no_such_wheels_dir"
    assert not bogus.exists()
    monkeypatch.setenv("AGENT_KB_VENDORED_DEPS", str(bogus))

    with pytest.raises(SystemExit) as excinfo:
        launcher._validate_vendored_deps()

    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    payloads = _stderr_payload_lines(captured.err)
    assert len(payloads) == 1
    payload = payloads[0]

    assert payload["error"] == "vendored_deps_invalid"
    assert payload["path"] == str(bogus)
    assert "AGENT_KB_VENDORED_DEPS" in payload["detail"]
    assert payload["spec_id"] == launcher._SPEC_ID
