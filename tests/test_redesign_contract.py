"""
Redesign contract tests (SCAFFOLD STUBS — all skipped).

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        success_criteria + architecture.components

These tests pin the v0.6.0 invariants that the redesign MUST NOT regress.
Each test is currently `@pytest.mark.skip(...)` with a TODO docstring
describing the assertion. Phase 0 / Phase 2 implementations promote these
stubs into real assertions.

The four invariants:

  1. test_stderr_log_schema_pinned
     Every knowledgebase_stderr_log emission has the 11 fields exactly
     (kb_id, op, phase, elapsed_ms, rows_in, rows_ok, rows_skipped,
     rows_failed, dedup_policy, request_id, tool_caller_version).

  2. test_probe4_kb_info_response_shape
     kb_info returns the fields probe-4 inspects (source_type, uri,
     dedup_key, page_id, dominant_embedding_model). Backends MUST return
     null for inapplicable fields, NOT omit them.

  3. test_mcp_tool_surface_unchanged
     The set of @mcp.tool()-decorated functions equals the v0.6.0 list of
     25 tools — empirically confirmed at server.py decorator hits on lines
     164,180,195,225,265,280,295,313,328,343,364,376,393,416,433,450,482,
     499,514,532,602,628,695,724,767.

  4. test_settings_resolve_paths_produces_friendly_error
     Settings().resolve_paths() with unset AGENT_KB_SAVES_DIR produces a
     STRUCTURED config_missing payload, not a raw pydantic.ValidationError.
     This is the Bug-2 fix per validation finding f-06.
"""

from __future__ import annotations

import pytest

# The 25 frozen kb_* MCP tools per spec.architecture.components[3] and
# overview.objectives[4], PLUS the additive `kb_migrate` introduced by
# Phase 4 of the v2.1 redesign (probe-4-safe — adds a new tool, does
# NOT regress any existing one). Set is now 26 names.
#
# Any redesign change that adds or removes from this list MUST fail the
# contract test below. Subsequent phases that introduce additional
# additive tools should extend this set (NEVER shrink it). The 25 names
# below are pinned by spec; ``kb_migrate`` is the first Phase 4 add.
_FROZEN_MCP_TOOLS_V0_6_0 = frozenset(
    {
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
        # Phase 4 additive (spec.implementation.phases[4]):
        "kb_migrate",
    }
)

# The 11 frozen fields of every knowledgebase_stderr_log emission per
# spec.architecture.integration_points and v0.6.0 CHANGELOG.
_STDERR_LOG_REQUIRED_FIELDS = frozenset(
    {
        "kb_id",
        "op",
        "phase",
        "elapsed_ms",
        "rows_in",
        "rows_ok",
        "rows_skipped",
        "rows_failed",
        "dedup_policy",
        "request_id",
        "tool_caller_version",
    }
)

# Probe-4 minimum response shape per spec.constraints[4] and risks[0]. Backends
# MUST return null for inapplicable fields, NOT omit them.
_PROBE4_KB_INFO_REQUIRED_FIELDS = frozenset(
    {
        "source_type",
        "uri",
        "dedup_key",
        "page_id",
        "dominant_embedding_model",
    }
)


@pytest.mark.skip(reason="Phase 0 — implement when bug-fix work begins")
def test_stderr_log_schema_pinned() -> None:
    """
    TODO[Phase 0/Phase 2]: Capture every knowledgebase_stderr_log emission
    during a representative kb_ingest_batch run and assert the JSON payload
    contains EXACTLY the 11 required fields (no extras, no omissions).

    Implementation sketch:
      - monkeypatch services/stderr_log.py to redirect stderr to a list
      - run kb_create + kb_ingest with a fixture source
      - assert each emitted line is JSON
      - assert json.loads(line).keys() == _STDERR_LOG_REQUIRED_FIELDS
      - extra fields beyond the 11 fail the test (probe-4 stability)

    Reference: spec.architecture.integration_points[2]
               (knowledgebase_stderr_log 11-field schema preserved across
               backends; pinned by tests/contract/test_stderr_schema.py)
    """
    raise NotImplementedError("Stub — Phase 0/Phase 2 deliverable")


@pytest.mark.skip(reason="Phase 0 — implement when bug-fix work begins")
def test_probe4_kb_info_response_shape() -> None:
    """
    TODO[Phase 0/Phase 2]: Call kb_info on a fixture KB and assert the
    response JSON contains AT LEAST the 5 probe-4 fields. Inapplicable
    fields must return None (not be omitted).

    Implementation sketch:
      - create kb_id under fixture saves_dir
      - kb_ingest a fixture file
      - call kb_info
      - response = json.loads(resp)
      - for field in _PROBE4_KB_INFO_REQUIRED_FIELDS:
            assert field in response
      - assert chromadb backend populates all fields
      - Phase 3 extension: assert markdown backend returns
            dominant_embedding_model=None (NOT missing)

    Reference: spec.risks[0] mitigation; success_criteria[4]; validation
               finding f-20 (probe-4 contract preservation).
    """
    raise NotImplementedError("Stub — Phase 0/Phase 2 deliverable")


@pytest.mark.skip(reason="Phase 0 — implement when bug-fix work begins")
def test_mcp_tool_surface_unchanged() -> None:
    """
    TODO[Phase 0/Phase 2]: Introspect mcp._tool_manager._tools after server
    import and assert the set of registered tool names equals the frozen
    v0.6.0 list (25 tools).

    Implementation sketch:
        from agent_knowledgebase.server import mcp
        registered = set(mcp._tool_manager._tools.keys())
        assert registered == _FROZEN_MCP_TOOLS_V0_6_0

    NOTE: Phase 4 introduces the additive kb_migrate tool. At that point,
    extend _FROZEN_MCP_TOOLS_V0_6_0 to add 'kb_migrate' — DO NOT remove any
    existing tool.

    Reference: spec.overview.objectives[4]; success_criteria[9]; empirical
               grep at server.py decorator hits on lines 164,180,195,225,
               265,280,295,313,328,343,364,376,393,416,433,450,482,499,514,
               532,602,628,695,724,767. validation finding f-07 confirmed
               25 tools verbatim.
    """
    raise NotImplementedError("Stub — Phase 0/Phase 2 deliverable")


@pytest.mark.skip(reason="Phase 0 — implement when bug-fix work begins")
def test_settings_resolve_paths_produces_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    TODO[Phase 0]: Bug-2 fix — monkey-patch os.environ to remove
    AGENT_KB_SAVES_DIR; call kb_list (or any kb_* tool) via FastMCP; assert
    response is a structured JSON {"error":"config_missing",
    "missing":"AGENT_KB_SAVES_DIR", ...} payload, NOT a raw
    pydantic.ValidationError propagating as opaque MCP InternalError.

    Implementation sketch:
        monkeypatch.delenv('AGENT_KB_SAVES_DIR', raising=False)
        # also clear any user/project JSON config under fixture HOME
        from agent_knowledgebase.server import mcp
        kb_list = mcp._tool_manager._tools['kb_list'].fn
        result = kb_list()  # MCP-style invocation
        payload = json.loads(result)
        assert payload == {
            'error': 'config_missing',
            'missing': 'AGENT_KB_SAVES_DIR',
            'detail': '...'
        }

    Phase 0 also adds a default_factory to config.py saves_dir Field so
    that ~/.agent-kb/saves is the fallback when no env var is set. Once
    that ships, this test asserts that EITHER (a) the fallback path is
    used and kb_list returns an empty list, OR (b) the structured error
    payload is returned if the fallback path is unwritable.

    THIS IS THE CORRECTED Bug-2 NARRATIVE per v2.1 spec revision. The
    v2.0 'decorator NO-OP' framing was inverted — the current decorator
    order (@mcp.tool() outer, @_with_tool_timeout inner) works
    empirically (tests/test_tool_timeout.py 6/6 passing). DO NOT swap
    decorators. See docs/redesign/MISTAKES.md M-01.

    Reference: validation finding f-06 (root cause); spec.notes FINAL Bug-2
               fix list; spec.implementation.phases[0].tasks[0-2]
               (wrap _get_service + saves_dir default_factory + main()
               startup precheck).
    """
    raise NotImplementedError("Stub — Phase 0 deliverable")
