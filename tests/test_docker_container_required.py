"""bin/run_server.py: container detection tests.

These tests verify the Docker shim correctly detects whether the
agent-knowledgebase container is running, without invoking real Docker.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

# Module under test imports docker_shim from bin/run_server via runpy in
# integration tests. For the unit-test layer, we exercise the helpers
# directly via a side-import of the script.
import sys
from pathlib import Path

_BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
sys.path.insert(0, str(_BIN_DIR))
import run_server  # type: ignore[import-not-found]


def test_is_container_running_true_when_inspect_returns_running(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = '"running"\n'
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    assert run_server.is_container_running("agent-knowledgebase") is True


def test_is_container_running_false_when_status_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = '"exited"\n'
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    assert run_server.is_container_running("agent-knowledgebase") is False


def test_is_container_running_false_when_inspect_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.returncode = 1
    fake.stdout = ""
    fake.stderr = "Error: No such container: agent-knowledgebase\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    assert run_server.is_container_running("agent-knowledgebase") is False
