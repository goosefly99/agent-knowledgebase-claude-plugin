#!/usr/bin/env python3
"""
agent-knowledgebase Pattern A launcher.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        architecture.components[0] "bin/run_server.py — Pattern A single Python launcher"

This file is the entry-point referenced by .mcp.json:

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

Behaviour (Phase 1, v0.8.0):

  1. Refuse Python<3.11 (structured stderr `python_version`).
  2. Resolve `${CLAUDE_PLUGIN_ROOT}` (or fall back to `<file>.parent.parent`).
     Cygwin / Git-Bash quirk: if `sys.platform=='win32'` AND `MSYSTEM` is set,
     forward-slash paths are normalised via `Path(...).resolve()`.
  3. Verify the plugin root is writable (structured stderr
     `read_only_filesystem`).
  4. Acquire a stdlib-only cross-process lock on
     `<plugin_root>/.venv.lock` (`fcntl.flock` on Unix, `msvcrt.locking`
     on Windows) so concurrent worker spawns serialise the bootstrap.
     We do NOT depend on the pip `filelock` package: this runs BEFORE pip.
  5. Create `<plugin_root>/.venv` via stdlib `venv` if missing.
  6. SHA256 of `<plugin_root>/requirements.lock` is compared against
     `<plugin_root>/.venv/.req-sha`. On match, skip pip install (fast path).
     On mismatch, run `pip install --timeout 30 -r requirements.lock`
     in the venv. `subprocess.TimeoutExpired` is surfaced as structured
     stderr `network_unreachable`.
  7. `AGENT_KB_VENDORED_DEPS=/path/to/wheels` swaps in
     `--no-index --find-links=$AGENT_KB_VENDORED_DEPS` for air-gapped
     installs.
  8. After successful install, write the new sentinel.
  9. `os.execv` to `<venv>/bin/python` (POSIX) or
     `<venv>\\Scripts\\python.exe` (Windows) with
     `["-m", "agent_knowledgebase.server"]` so this process is replaced
     and does not stay resident.

NO bash/cmd companion. Single .py file is the design.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

# Spec_id pinned for traceability — future agents grep this comment to find
# the launcher that implements the v2.1 Pattern A design.
_SPEC_ID = "70ab2170-381a-4657-bcd1-28a40c6f369b"
_SPEC_VERSION = "2.1"

# Wall-clock budget for the bootstrap pip install. PyPI is fast; 30 s is
# enough for a captive-portal user to notice their network is unreachable
# without making the launcher feel hung.
_PIP_INSTALL_TIMEOUT_SECONDS = 30


def _emit_stderr_error(error_token: str, detail: str, **extra: object) -> None:
    """
    Emit a structured single-line JSON error to stderr.

    The three canonical error_token values per spec are:
      - ``python_version``        Python<3.11 detected
      - ``network_unreachable``   PyPI 30 s timeout — set AGENT_KB_VENDORED_DEPS
      - ``read_only_filesystem``  CLAUDE_PLUGIN_ROOT not writable
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
    ``<this_file>.parent.parent`` so the launcher works in dev when
    invoked directly via ``python bin/run_server.py``.

    Cygwin / Git-Bash quirk: on Windows, the env var may contain forward
    slashes ("C:/Users/.../plugin"). ``Path(...).resolve()`` collapses
    these to backslashes via the host OS rules, so subprocess later
    receives a Windows-shaped path. ``MSYSTEM`` is set under
    Git-Bash / MSYS2 / Cygwin shells and is the canonical detection
    signal.
    """
    env_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if env_root:
        # Path(...).resolve() normalises mixed slashes per the host OS,
        # which is what subprocess on Windows wants. Explicit MSYSTEM
        # check kept for grep-ability (per spec task list).
        if sys.platform == "win32" and os.environ.get("MSYSTEM"):
            return Path(env_root).resolve()
        return Path(env_root).resolve()
    return Path(__file__).resolve().parent.parent


def _resolve_venv_python(plugin_root: Path) -> Path:
    """
    Compute the path to the venv's python interpreter.

    Windows: ``<plugin_root>/.venv/Scripts/python.exe``
    Unix:    ``<plugin_root>/.venv/bin/python``
    """
    venv_dir = plugin_root / ".venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _check_python_version() -> None:
    """Fail-fast if Python<3.11. Required floor per spec assumptions."""
    if sys.version_info < (3, 11):
        _emit_stderr_error(
            "python_version",
            f"Python>=3.11 required; detected "
            f"{sys.version_info.major}.{sys.version_info.minor}",
            detected=f"{sys.version_info.major}.{sys.version_info.minor}",
            required=">=3.11",
        )
        sys.exit(1)


def _check_writable(plugin_root: Path) -> None:
    """Refuse to bootstrap on a read-only filesystem.

    ``os.access(..., os.W_OK)`` matches the mitigation called out in the
    spec (`read_only_filesystem` token). The structured stderr payload
    points the user at ``AGENT_KB_VENDORED_DEPS`` and at writing the
    venv to a different host-mounted directory.
    """
    if not plugin_root.exists():
        # The directory itself is missing — not strictly read-only but
        # equally unbootstrappable. Surface the same token so the user
        # sees a single recovery hint.
        _emit_stderr_error(
            "read_only_filesystem",
            f"CLAUDE_PLUGIN_ROOT does not exist: {plugin_root}",
            plugin_root=str(plugin_root),
        )
        sys.exit(1)
    if not os.access(plugin_root, os.W_OK):
        _emit_stderr_error(
            "read_only_filesystem",
            f"CLAUDE_PLUGIN_ROOT is not writable: {plugin_root}. "
            "Mount it writable or relocate ${CLAUDE_PLUGIN_ROOT} to a "
            "writable directory.",
            plugin_root=str(plugin_root),
        )
        sys.exit(1)


@contextlib.contextmanager
def _bootstrap_lock(lock_path: Path):
    """Stdlib-only cross-process lock on ``<plugin_root>/.venv.lock``.

    On Unix uses ``fcntl.flock`` (advisory, blocking). On Windows uses
    ``msvcrt.locking`` with a tiny retry loop because ``msvcrt.locking``
    locks at most ``LK_LOCK`` blocks and raises ``OSError`` with EDEADLK
    if it cannot acquire after ~10 s. We sidestep the dependency on the
    pip ``filelock`` package because this code path runs BEFORE pip
    install.

    The lock file itself is a 1-byte placeholder; we never read or write
    its contents. We only use the OS-level lock on the open handle.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure the file exists; opening with mode "a+" both creates and
    # gives us a writable handle without truncating any prior placeholder.
    fh = open(lock_path, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            # msvcrt.locking requires a non-empty region. Write a sentinel
            # byte if the file is empty so locking has something to grab.
            if os.fstat(fh.fileno()).st_size == 0:
                fh.write("0")
                fh.flush()
            fh.seek(0)
            # Retry up to 60 s in 1 s slices — concurrent worker spawns
            # should always make progress within the bootstrap window.
            import time

            deadline = time.monotonic() + 60.0
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.5)
            try:
                yield
            finally:
                # Release the same byte range we locked.
                fh.seek(0)
                with contextlib.suppress(OSError):
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def _compute_lock_sha(requirements_path: Path) -> str | None:
    """SHA256 of requirements.lock contents, or None if the file is missing.

    A missing requirements.lock is treated as a fresh install — nothing
    to compare against, so the bootstrap will run pip from scratch.
    """
    if not requirements_path.exists():
        return None
    h = hashlib.sha256()
    with open(requirements_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sentinel(sentinel_path: Path) -> str | None:
    """Read the cached SHA from ``<venv>/.req-sha``, or None if missing."""
    if not sentinel_path.exists():
        return None
    try:
        return sentinel_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _write_sentinel(sentinel_path: Path, sha: str) -> None:
    """Persist the new SHA after a successful pip install."""
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text(sha, encoding="utf-8")


def _ensure_venv(plugin_root: Path, venv_python: Path) -> None:
    """Create ``<plugin_root>/.venv`` via stdlib ``venv`` if missing."""
    if venv_python.exists():
        return
    venv_dir = plugin_root / ".venv"
    # with_pip=True bootstraps pip into the new venv so the install step
    # below has something to invoke.
    venv.create(str(venv_dir), with_pip=True, clear=False, symlinks=False)


def _build_pip_install_argv(
    venv_python: Path, requirements_path: Path
) -> list[str]:
    """Build the pip-install argv, honouring ``AGENT_KB_VENDORED_DEPS``.

    With the env var set, ``--no-index --find-links=<dir>`` makes pip
    skip PyPI entirely so air-gapped corporate hosts can preload wheels
    and bootstrap without network access. Without it, the launcher
    relies on the public PyPI mirror.
    """
    argv: list[str] = [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--timeout",
        str(_PIP_INSTALL_TIMEOUT_SECONDS),
    ]
    vendored = os.environ.get("AGENT_KB_VENDORED_DEPS")
    if vendored:
        argv.extend(["--no-index", "--find-links", vendored])
    argv.extend(["-r", str(requirements_path)])
    return argv


def _run_pip_install(
    venv_python: Path, requirements_path: Path
) -> None:
    """Run pip install inside the venv with a 30 s wall-clock budget.

    A ``TimeoutExpired`` is surfaced as structured stderr
    ``network_unreachable``; any other non-zero exit re-raises so the
    user sees the underlying pip output (which itself is informative).
    """
    argv = _build_pip_install_argv(venv_python, requirements_path)
    try:
        subprocess.run(
            argv,
            check=True,
            timeout=_PIP_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _emit_stderr_error(
            "network_unreachable",
            f"pip install exceeded {_PIP_INSTALL_TIMEOUT_SECONDS}s. "
            "Set AGENT_KB_VENDORED_DEPS=/path/to/wheels for air-gapped "
            "installs (vendor wheels via "
            "`pip download -d wheels/ -r requirements.lock`).",
            timeout_seconds=_PIP_INSTALL_TIMEOUT_SECONDS,
        )
        sys.exit(1)


def _bootstrap_venv(plugin_root: Path) -> Path:
    """Run the full bootstrap and return the venv python path.

    Idempotent: a second call after a successful first run is the
    fast path — the SHA matches and we skip the pip install entirely.
    """
    venv_python = _resolve_venv_python(plugin_root)
    requirements_path = plugin_root / "requirements.lock"
    sentinel_path = plugin_root / ".venv" / ".req-sha"
    lock_path = plugin_root / ".venv.lock"

    with _bootstrap_lock(lock_path):
        # (1) ensure the venv exists.
        _ensure_venv(plugin_root, venv_python)

        # (2) compute desired vs cached SHA.
        desired_sha = _compute_lock_sha(requirements_path)
        cached_sha = _read_sentinel(sentinel_path)

        # If there's no requirements.lock at all, we have nothing to install
        # and nothing to validate. Most likely a developer-clone scenario
        # where dependencies were installed manually; fall through to execv.
        if desired_sha is None:
            return venv_python

        # (3) fast path — sentinel matches, skip pip install.
        if cached_sha == desired_sha:
            return venv_python

        # (4) slow path — install and persist sentinel.
        _run_pip_install(venv_python, requirements_path)
        _write_sentinel(sentinel_path, desired_sha)

    return venv_python


def main() -> None:
    """Pattern A launcher entry point."""
    _check_python_version()

    plugin_root = _resolve_plugin_root()
    _check_writable(plugin_root)

    venv_python = _bootstrap_venv(plugin_root)

    # Defensive: if for some reason the venv python still does not
    # exist (e.g. ensure_venv silently failed because the host is mid
    # filesystem teardown), fall back to the running interpreter so we
    # at least produce a meaningful import error rather than ENOENT.
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    args = [str(venv_python), "-m", "agent_knowledgebase.server"]
    os.execv(args[0], args)


if __name__ == "__main__":
    main()
