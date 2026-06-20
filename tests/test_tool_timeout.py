"""Tests: every MCP tool is bounded by a wall-clock timeout.

Prevents a stalled embedder, ingestor, or DB call from blocking the MCP
RPC indefinitely.  Covers the ``_with_tool_timeout`` decorator in
``agent_knowledgebase.server``:

* a slow tool returns a ``tool_timeout`` JSON payload (does not raise);
* a fast tool returns its normal string result;
* an exception raised inside a tool still propagates to the caller;
* ``inspect.signature`` still sees the original tool's parameters so
  FastMCP can build its tool schema from the wrapped function;
* after a timeout, a subsequent call is NOT wedged behind the stuck
  worker — the executor is reset so the new call runs on a fresh thread.
"""

from __future__ import annotations

import inspect
import json
import threading
import time

import pytest

from agent_knowledgebase import server


@pytest.fixture()
def short_timeout(monkeypatch: pytest.MonkeyPatch) -> float:
    """Shrink ``_TOOL_TIMEOUT_SECONDS`` to 0.2s so tests don't wait 3 minutes.

    Returns the applied value so tests can assert it shows up in payloads.
    """
    value = 0.2
    monkeypatch.setattr(server, "_TOOL_TIMEOUT_SECONDS", value)
    return value


@pytest.fixture(autouse=True)
def _reset_executor_between_tests() -> None:
    """Start every test with a fresh tool executor and no cached service."""
    server._reset_tool_executor()
    yield
    server._reset_tool_executor()


def test_slow_tool_returns_tool_timeout_payload(short_timeout: float) -> None:
    @server._with_tool_timeout
    def slow_tool(x: int) -> str:
        time.sleep(2.0)  # much longer than short_timeout
        return str(x)

    payload = json.loads(slow_tool(42))
    assert payload == {
        "error": "tool_timeout",
        "tool": "slow_tool",
        "timeout_seconds": short_timeout,
    }


def test_fast_tool_returns_normal_result(short_timeout: float) -> None:
    @server._with_tool_timeout
    def fast_tool(x: int) -> str:
        return f"ok:{x}"

    assert fast_tool(7) == "ok:7"


def test_exception_propagates_to_caller(short_timeout: float) -> None:
    @server._with_tool_timeout
    def boom_tool() -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        boom_tool()


def test_signature_preserved_for_fastmcp(short_timeout: float) -> None:
    """FastMCP reads tool schemas via ``inspect.signature`` — ``__wrapped__``
    must survive so the MCP tool JSON schema still names real parameters.
    """
    def original(kb_id: str, top_k: int = 10) -> str:
        return ""

    wrapped = server._with_tool_timeout(original)
    # inspect.signature follows __wrapped__ by default, so the wrapped
    # function must advertise the original parameter list and defaults.
    assert inspect.signature(wrapped) == inspect.signature(original)
    assert list(inspect.signature(wrapped).parameters) == ["kb_id", "top_k"]
    assert inspect.signature(wrapped).parameters["top_k"].default == 10


def test_timeout_does_not_wedge_subsequent_calls(short_timeout: float) -> None:
    """After a timeout the executor is reset; the next call must complete.

    Regression guard: if we kept the old single-worker executor, the next
    call would queue behind the stuck task and also time out, leaving the
    server permanently wedged.
    """
    released = threading.Event()

    @server._with_tool_timeout
    def stuck_tool() -> str:
        released.wait(timeout=5.0)  # released by the test's finally clause
        return "eventually"

    @server._with_tool_timeout
    def quick_tool() -> str:
        return "fresh"

    try:
        first = json.loads(stuck_tool())
        assert first["error"] == "tool_timeout"

        # A follow-up call on a fresh worker must succeed immediately.
        assert quick_tool() == "fresh"
    finally:
        # Unblock the abandoned worker so the daemon thread can exit cleanly.
        released.set()


def test_every_registered_mcp_tool_is_wrapped() -> None:
    """Regression guard: every ``@mcp.tool()`` in server.py must also carry
    ``@_with_tool_timeout``.  Checks via the ``__wrapped__`` chain.
    """
    from agent_knowledgebase.server import mcp

    registered = dict(mcp._tool_manager._tools)
    assert registered, "expected at least one tool registered on mcp"
    for name, tool in registered.items():
        fn = tool.fn
        # Walk the __wrapped__ chain; our decorator uses functools.wraps so
        # the original function is exactly one hop away.
        assert hasattr(fn, "__wrapped__"), (
            f"tool {name!r} is not wrapped — missing @_with_tool_timeout?"
        )
