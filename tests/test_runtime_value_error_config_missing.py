"""Phase 5 (I-06) — runtime ValueError("AGENT_KB_*") wrapped as config_missing.

After the Phase 5 default flip to ``provider='remote'``, an operator
on a fresh install with no ``AGENT_KB_EMBED_API_KEY`` would hit
``ValueError("AGENT_KB_EMBED_API_KEY required for remote
embeddings")`` raised inside the embedder build path on first
``kb_query``. The Phase 0 ``_classify_config_error`` only handled
``Settings.resolve_paths()`` errors; the runtime variant needs the
same wrap. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
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


def _scrub_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every env var that could supply embedder config."""
    for var in (
        "AGENT_KB_EMBED_API_KEY",
        "AGENT_KB_EMBED_BASE_URL",
        "AGENT_KB_EMBEDDING_PROVIDER",
        "AGENT_KB_EMBEDDING_MODEL",
        "AGENT_KB_USER_CONFIG",
        "AGENT_KB_PROJECT_CONFIG",
        "AGENT_KB_PROJECT_DIR",
        "CLAUDE_PROJECT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)


def test_kb_query_returns_config_missing_when_remote_api_key_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh install + post-Phase-5 default + no API key →
    structured ``config_missing`` payload, NOT a stack trace.

    Mirrors the ``test_get_service_config_missing.py`` pattern: invoke
    the registered FastMCP callable directly; assert the JSON shape.
    """
    _scrub_config_env(monkeypatch)
    saves = tmp_path / "saves"
    saves.mkdir()
    monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))

    # Create a KB so we have something to query against. (Use the
    # service directly to keep this independent of kb_create's
    # success/failure paths.)
    svc = server._get_service()
    kb = svc.create_kb(name="cfg-missing")

    # Insert a chunk so the snapshot returns a model that the post-flip
    # remote default would try to query under, triggering the
    # ValueError on first kb_query.
    import json as _json
    ctx = svc._ctx(kb.id)
    ctx.db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"src-{kb.id}", kb.id, "file", "x://x", "ingested"),
    )
    ctx.db._conn.execute(
        "INSERT INTO chunks (id, source_id, kb_id, content, metadata, "
        "embedding_id, embedding_provider, embed_base_url, embedder_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "ch1",
            f"src-{kb.id}",
            kb.id,
            "test content",
            _json.dumps({
                "embedding_model": "text-embedding-3-small",
                "embedding_provider": "remote",
                "embed_base_url": "https://api.openai.com/v1",
            }),
            None,
            "remote",
            "https://api.openai.com/v1",
            "remote/text-embedding-3-small@https://api.openai.com/v1",
        ),
    )
    ctx.db._conn.commit()

    # Invoke the registered kb_query through the MCP boundary so the
    # _with_tool_timeout wrap engages.
    fn = server.mcp._tool_manager._tools["kb_query"].fn
    result = fn(kb_id=kb.id, text="anything", top_k=5)
    payload = json.loads(result)

    assert isinstance(payload, dict), (
        f"kb_query must return a single config_missing object, not a list; "
        f"got {payload!r}"
    )
    assert payload.get("error") == "config_missing", (
        f"Expected error='config_missing'; got {payload!r}"
    )
    assert payload.get("missing") == "AGENT_KB_EMBED_API_KEY", (
        f"Expected missing='AGENT_KB_EMBED_API_KEY'; got {payload!r}"
    )
    assert payload.get("detail"), "config_missing payload must include detail"


def test_unrelated_value_error_still_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrap must NOT swallow ValueErrors with messages that don't
    match the AGENT_KB_<VAR> required pattern — those are real bugs
    and should still surface.
    """
    _scrub_config_env(monkeypatch)
    saves = tmp_path / "saves"
    saves.mkdir()
    monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
    monkeypatch.setenv("AGENT_KB_EMBED_API_KEY", "sk-test")

    # kb_query against a non-existent kb_id raises
    # ValueError("KB <kb_id> not found") inside the service.
    fn = server.mcp._tool_manager._tools["kb_query"].fn
    with pytest.raises(ValueError, match="not found"):
        fn(kb_id="does-not-exist", text="x", top_k=1)
