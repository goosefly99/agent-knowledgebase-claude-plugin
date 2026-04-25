"""Tests for the Pattern A launcher's bootstrap path.

Spec: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
      implementation.phases[1] "Phase 1 — Pattern A single-launcher runtime
      bootstrap". Specifically the deliverable
      ``tests/test_run_server_bootstrap.py``.

These tests exercise ``bin/run_server.py``'s ``_bootstrap_venv`` helper —
the function that owns the venv create + sentinel SHA dance. We do NOT
exercise ``main()`` end-to-end because ``main()`` ends in ``os.execv``
which would replace the pytest process. Instead we drive the same code
path through the public helper, mocking out ``venv.create`` and
``subprocess.run`` to keep the tests fast and offline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Module loader — the launcher lives at bin/run_server.py, outside the
# importable src/ tree. We load it by file path so the tests don't need
# any sys.path gymnastics.
# -----------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LAUNCHER_PATH = _PROJECT_ROOT / "bin" / "run_server.py"


def _load_launcher_module():
    """Import bin/run_server.py as ``run_server`` for testing."""
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


# -----------------------------------------------------------------------------
# Helpers — standard fixture skeletons so each test reads top-down.
# -----------------------------------------------------------------------------


def _make_fake_venv(plugin_root: Path) -> Path:
    """Create a fake .venv layout matching the launcher's expectations.

    Used in tests where we want to claim the venv already exists so the
    launcher skips the ``venv.create`` step. Returns the path to the
    fake venv-python so tests can assert the launcher returns it.
    """
    venv_dir = plugin_root / ".venv"
    if sys.platform == "win32":
        scripts_dir = venv_dir / "Scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        venv_python = scripts_dir / "python.exe"
    else:
        bin_dir = venv_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        venv_python = bin_dir / "python"
    venv_python.write_text("#!/usr/bin/env python3\n")
    return venv_python


def _write_lock(plugin_root: Path, contents: str = "mcp==1.0.0\n") -> Path:
    """Write a stand-in requirements.lock under the fake plugin root."""
    lock_path = plugin_root / "requirements.lock"
    lock_path.write_text(contents)
    return lock_path


# -----------------------------------------------------------------------------
# Test 1 — fresh CLAUDE_PLUGIN_ROOT triggers venv create + sentinel write.
# -----------------------------------------------------------------------------


def test_fresh_plugin_root_creates_venv_and_writes_sentinel(
    launcher, tmp_path: Path
) -> None:
    """First-launch fast path: no venv, no sentinel — pip install runs once.

    We mock ``venv.create`` so the tests do not actually create a venv on
    disk (which would download pip and take seconds). After ``venv.create``
    is "called", we synthesise the venv-python file ourselves so the
    launcher's existence check passes.
    """
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    _write_lock(plugin_root, "mcp==1.0.0\n")

    venv_python = (
        (plugin_root / ".venv" / "Scripts" / "python.exe")
        if sys.platform == "win32"
        else (plugin_root / ".venv" / "bin" / "python")
    )

    def fake_venv_create(_path, **_kwargs):
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n")

    fake_subprocess_run = MagicMock()

    with patch.object(launcher.venv, "create", side_effect=fake_venv_create) as create_mock:
        with patch.object(launcher.subprocess, "run", fake_subprocess_run):
            returned = launcher._bootstrap_venv(plugin_root)

    # venv.create was invoked exactly once with the .venv path.
    assert create_mock.call_count == 1
    # pip install was invoked.
    assert fake_subprocess_run.called
    # The launcher returned the venv-python path.
    assert returned == venv_python
    # The sentinel was written.
    sentinel = plugin_root / ".venv" / ".req-sha"
    assert sentinel.exists()
    # And matches the lock SHA.
    assert sentinel.read_text(encoding="utf-8").strip() == launcher._compute_lock_sha(
        plugin_root / "requirements.lock"
    )


# -----------------------------------------------------------------------------
# Test 2 — second invocation skips pip install (sentinel matches).
# -----------------------------------------------------------------------------


def test_second_invocation_skips_pip_install(
    launcher, tmp_path: Path
) -> None:
    """Fast path: venv exists and sentinel SHA matches — pip is not invoked."""
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    lock_path = _write_lock(plugin_root, "mcp==1.0.0\n")

    # Pre-create the venv-python so launcher skips venv.create.
    venv_python = _make_fake_venv(plugin_root)

    # Pre-write the sentinel matching the lock SHA so launcher skips pip.
    sentinel = plugin_root / ".venv" / ".req-sha"
    sentinel.write_text(launcher._compute_lock_sha(lock_path), encoding="utf-8")

    fake_subprocess_run = MagicMock()
    with patch.object(launcher.venv, "create") as create_mock:
        with patch.object(launcher.subprocess, "run", fake_subprocess_run):
            returned = launcher._bootstrap_venv(plugin_root)

    # Neither venv.create nor subprocess.run was called.
    create_mock.assert_not_called()
    fake_subprocess_run.assert_not_called()
    # And the launcher still returned the venv-python path.
    assert returned == venv_python


# -----------------------------------------------------------------------------
# Test 3 — AGENT_KB_VENDORED_DEPS adds --no-index --find-links.
# -----------------------------------------------------------------------------


def test_vendored_deps_uses_offline_pip_argv(
    launcher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``AGENT_KB_VENDORED_DEPS`` is set, pip runs without PyPI."""
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    _write_lock(plugin_root, "mcp==1.0.0\n")

    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    monkeypatch.setenv("AGENT_KB_VENDORED_DEPS", str(wheels_dir))

    venv_python = (
        (plugin_root / ".venv" / "Scripts" / "python.exe")
        if sys.platform == "win32"
        else (plugin_root / ".venv" / "bin" / "python")
    )

    def fake_venv_create(_path, **_kwargs):
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/usr/bin/env python3\n")

    fake_subprocess_run = MagicMock()

    with patch.object(launcher.venv, "create", side_effect=fake_venv_create):
        with patch.object(launcher.subprocess, "run", fake_subprocess_run):
            launcher._bootstrap_venv(plugin_root)

    fake_subprocess_run.assert_called_once()
    argv = fake_subprocess_run.call_args.args[0]
    # The argv must contain the offline flags pointing at wheels_dir,
    # and must NOT have -i / --index-url overridden by something else.
    assert "--no-index" in argv
    assert "--find-links" in argv
    assert str(wheels_dir) in argv
    assert "-r" in argv
    assert str(plugin_root / "requirements.lock") in argv


# -----------------------------------------------------------------------------
# Bonus — argv builder is tested independently for documentation value.
# -----------------------------------------------------------------------------


def test_pip_argv_default_omits_offline_flags(
    launcher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default mode (no AGENT_KB_VENDORED_DEPS) talks to PyPI."""
    monkeypatch.delenv("AGENT_KB_VENDORED_DEPS", raising=False)
    fake_python = tmp_path / "venv-python"
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("mcp==1.0.0\n")

    argv = launcher._build_pip_install_argv(fake_python, lock_path)

    assert "--no-index" not in argv
    assert "--find-links" not in argv
    assert "--timeout" in argv
    assert "30" in argv
    assert str(lock_path) in argv


def test_pip_argv_with_vendored_deps_includes_offline_flags(
    launcher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Argv-level assertion: vendored deps swap in offline flags."""
    fake_python = tmp_path / "venv-python"
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("mcp==1.0.0\n")
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()
    monkeypatch.setenv("AGENT_KB_VENDORED_DEPS", str(wheels_dir))

    argv = launcher._build_pip_install_argv(fake_python, lock_path)

    assert "--no-index" in argv
    assert "--find-links" in argv
    # find-links arg follows --find-links immediately
    idx = argv.index("--find-links")
    assert argv[idx + 1] == str(wheels_dir)
