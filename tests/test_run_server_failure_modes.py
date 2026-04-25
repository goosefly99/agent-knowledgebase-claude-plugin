"""Tests for the three structured stderr error modes of the Pattern A launcher.

Spec: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
      implementation.phases[1] "Phase 1 — Pattern A single-launcher runtime
      bootstrap" — deliverable ``tests/test_run_server_failure_modes.py``.

Three structured-stderr error modes are surfaced by the launcher:

    {"error":"python_version",      "detected":"3.10", "required":">=3.11", ...}
    {"error":"network_unreachable", "timeout_seconds":30, ...}
    {"error":"read_only_filesystem","plugin_root":"...", ...}

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
