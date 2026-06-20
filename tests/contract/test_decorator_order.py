"""Regression guard for the @mcp.tool() / @_with_tool_timeout decorator order.

This test exists EXPLICITLY to prevent accidental future inversion of the
two decorators on every kb_* tool in ``server.py``. It does NOT mandate a
specific nesting order — only that every registered tool carries
``__wrapped__``, which means it has been wrapped by ``functools.wraps``
(i.e. by ``@_with_tool_timeout``) and that wrapping survived FastMCP
registration.

Why this matters
----------------

The current order is:

    @mcp.tool()           # OUTER
    @_with_tool_timeout   # INNER
    def kb_create(...): ...

Python evaluates decorators bottom-up:

1. ``_with_tool_timeout`` wraps ``kb_create`` first, producing a
   ``wrapper`` function whose ``__wrapped__`` attribute (set by
   ``functools.wraps``) points to the original ``kb_create``.
2. ``mcp.tool()`` then registers that wrapper with FastMCP.

This is the WORKING order. ``tests/test_tool_timeout.py`` proves it
empirically: 6/6 pass including ``test_every_registered_mcp_tool_is_wrapped``,
and a live timeout test on a slow tool returns the structured
``{"error":"tool_timeout", ...}`` payload at the MCP boundary.

The v2.0 spec draft asserted the OPPOSITE — that ``@mcp.tool()`` was
outside ``@_with_tool_timeout`` and therefore the timeout was dead code at
the MCP boundary, recommending a swap. That diagnosis was INVERTED. Had
the swap been applied, FastMCP would register the raw, un-wrapped
function and the timeout would become dead code — flipping a working
setup into a broken one.

If this test ever fails, DO NOT swap the decorators back. Read
``docs/redesign/MISTAKES.md`` M-01 first, then debug what actually
removed the wrap.
"""

from __future__ import annotations

import pytest


def test_every_registered_tool_carries_a_wrap() -> None:
    """Every ``mcp._tool_manager._tools`` entry must have ``__wrapped__``.

    ``functools.wraps`` (used inside ``_with_tool_timeout``) is what sets
    ``__wrapped__``. Its presence proves the registered function went
    through the timeout wrapper before FastMCP registered it.
    """
    from agent_knowledgebase.server import mcp

    registered = dict(mcp._tool_manager._tools)
    assert registered, "expected at least one tool registered on mcp"

    missing: list[str] = []
    not_callable: list[str] = []
    for name, tool in registered.items():
        fn = tool.fn
        if not hasattr(fn, "__wrapped__"):
            missing.append(name)
            continue
        if not callable(fn.__wrapped__):
            not_callable.append(name)

    assert not missing, (
        f"the following registered tools are not wrapped (missing "
        f"__wrapped__) — has the @_with_tool_timeout decorator been "
        f"removed or moved? {sorted(missing)}"
    )
    assert not not_callable, (
        f"the following registered tools have a non-callable "
        f"__wrapped__: {sorted(not_callable)}"
    )


def test_every_kb_tool_in_v0_6_0_surface_is_wrapped() -> None:
    """The 25 frozen v0.6.0 kb_* tools must each be wrapped.

    Spot-checks the same invariant against the explicit list, so a
    regression that lost a single tool's decorator (rather than all of
    them) is also surfaced clearly.
    """
    from agent_knowledgebase.server import mcp

    expected = {
        "kb_create",
        "kb_delete",
        "kb_ingest",
        "kb_ingest_batch",
        "kb_update_source",
        "kb_remove_source",
        "kb_lint",
        "kb_lint_fix",
        "kb_rebuild_index",
        "kb_export",
        "kb_list",
        "kb_info",
        "kb_query",
        "kb_search",
        "kb_get_page",
        "kb_list_pages",
        "kb_get_source",
        "kb_list_sources",
        "kb_get_links",
        "kb_pipeline_status",
        "kb_config_path",
        "kb_config_show",
        "kb_config_get",
        "kb_config_set",
        "kb_config_validate",
    }

    registered = dict(mcp._tool_manager._tools)
    missing_from_registered = expected - set(registered)
    assert not missing_from_registered, (
        f"v0.6.0 frozen kb_* tools missing from registration: "
        f"{sorted(missing_from_registered)}"
    )

    not_wrapped = [
        name
        for name in expected
        if not hasattr(registered[name].fn, "__wrapped__")
    ]
    if not_wrapped:
        pytest.fail(
            f"the following v0.6.0 frozen kb_* tools lost their "
            f"@_with_tool_timeout wrap: {sorted(not_wrapped)}"
        )
