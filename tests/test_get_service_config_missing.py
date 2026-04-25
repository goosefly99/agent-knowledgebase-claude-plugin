"""Phase 0 Bug-2: ``_get_service`` returns a structured config-missing payload.

Validation finding f-06 confirmed the user-visible Bug-2: when
``AGENT_KB_SAVES_DIR`` is unset (and no JSON config supplies it),
``Settings().resolve_paths()`` raises ``pydantic.ValidationError`` /
``FileNotFoundError`` / ``NotADirectoryError`` and ``_get_service`` lets
that propagate as an opaque MCP InternalError on every ``kb_*`` call.

The Phase 0 fix wraps the body in a try/except, returning a sentinel
``_ConfigMissingService`` whose every method returns the JSON string
``{"error": "config_missing", "missing": "<env_var>", "detail": "..."}``.
This file pins three behaviours:

1. **Failure path** — when both ``AGENT_KB_SAVES_DIR`` is unset AND the
   ``~/.agent-kb/saves`` default-factory fallback also fails, ``kb_list``
   returns the structured ``config_missing`` JSON payload.
2. **Default-factory happy path** — when ``AGENT_KB_SAVES_DIR`` is unset
   but the new default-factory directory exists and is writable,
   ``kb_list`` returns ``[]`` and the wrap is invisible.
3. **Explicit env happy path** — when ``AGENT_KB_SAVES_DIR`` points at a
   valid directory, ``kb_list`` returns ``[]`` (proving the wrap does
   not break the existing flow).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_knowledgebase import server


@pytest.fixture(autouse=True)
def _reset_service_between_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a fresh, un-cached service singleton."""
    monkeypatch.setattr(server, "_service", None)
    server._reset_tool_executor()
    yield
    monkeypatch.setattr(server, "_service", None)
    server._reset_tool_executor()


def _invoke_kb_list() -> str:
    """Invoke the registered FastMCP ``kb_list`` callable and return its result."""
    fn = server.mcp._tool_manager._tools["kb_list"].fn
    return fn()


def _scrub_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every env var that could supply saves_dir or extra config.

    Includes the ``AGENT_KB_USER_CONFIG`` / ``AGENT_KB_PROJECT_CONFIG``
    overrides set by the project-wide ``conftest.py`` autouse fixture
    so that JSON config discovery falls back to defaults that don't
    exist in the fixture HOME.
    """
    for var in (
        "AGENT_KB_SAVES_DIR",
        "AGENT_KB_EXPORT_PATH",
        "AGENT_KB_USER_CONFIG",
        "AGENT_KB_PROJECT_CONFIG",
        "AGENT_KB_PROJECT_DIR",
        "CLAUDE_PROJECT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


def test_kb_list_returns_config_missing_payload_when_saves_dir_unresolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug-2 primary regression: opaque MCP error → structured payload.

    Even with the new default-factory in place, the resolved fallback
    path must STILL not exist on disk for this test to exercise the
    catch — otherwise the happy path takes over. We point ``Path.home``
    at an empty ``tmp_path`` so ``~/.agent-kb/saves`` resolves to a
    non-existent subdirectory.
    """
    _scrub_config_env(monkeypatch)
    # Point Path.home at tmp_path so the default-factory fallback
    # `~/.agent-kb/saves` resolves to a non-existent directory.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    # Sanity: the path the default-factory will produce does not exist.
    assert not (tmp_path / ".agent-kb" / "saves").exists()

    result = _invoke_kb_list()
    payload = json.loads(result)

    assert payload["error"] == "config_missing"
    assert payload["missing"] == "AGENT_KB_SAVES_DIR"
    assert "detail" in payload
    assert payload["detail"]


def test_kb_list_succeeds_when_default_factory_directory_is_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new default-factory makes a fresh install boot with no env vars."""
    _scrub_config_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    # Pre-create the default-factory target so resolve_paths()
    # accepts it.
    (tmp_path / ".agent-kb" / "saves").mkdir(parents=True)

    result = _invoke_kb_list()
    payload = json.loads(result)

    assert isinstance(payload, list)
    assert payload == []


def test_kb_list_succeeds_with_explicit_saves_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy-path proof that the try/except wrap does not regress success."""
    _scrub_config_env(monkeypatch)
    saves = tmp_path / "saves"
    saves.mkdir()
    monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))

    result = _invoke_kb_list()
    payload = json.loads(result)

    assert payload == []
