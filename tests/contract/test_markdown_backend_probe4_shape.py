"""Phase 3 contract test: probe-4 response shape preserved by the
markdown backend.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        constraints[4] (probe-4 minimum response shape frozen)
        + risks[0] mitigation (validation finding f-20)

The data-etl-orchestrator >= v0.4.0 plugin's probe-4 introspection
inspects the union of five frozen keys::

    source_type, uri, dedup_key, page_id, dominant_embedding_model

For the markdown backend ``dominant_embedding_model`` is ALWAYS ``None``
(markdown does not embed) — but the key MUST be present, NOT omitted.
This file pins that contract on
:meth:`MarkdownWikiBackend.info` directly. The MCP-level guard for
``kb_info`` / ``kb_list_pages`` / ``kb_list_sources`` shapes is
exercised by ``tests/contract/test_probe4_response_shape.py``; this
file is the backend-side companion that stays green even when the
service / MCP plumbing is mocked out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.backends.markdown_backend import MarkdownWikiBackend
from agent_knowledgebase.config import Settings


_PROBE4_BACKEND_INFO_KEYS = {
    "source_type",
    "uri",
    "dedup_key",
    "page_id",
    "dominant_embedding_model",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(tmp_path: Path) -> MarkdownWikiBackend:
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(saves_dir=saves, kb_backend="markdown").resolve_paths()
    return MarkdownWikiBackend(settings)


def _ingest_one(backend: MarkdownWikiBackend, kb_id: str) -> None:
    """Helper — write one canonical document so info() has data."""
    backend.index(
        kb_id=kb_id,
        documents=[
            {
                "id": "src-probe4",
                "content": "probe-4 markdown-backend fixture content",
                "metadata": {
                    "source_type": "file",
                    "title": "Probe Four Fixture",
                    "uri": "/tmp/probe4.txt",
                    "dedup_key": "probe4-key",
                },
            }
        ],
    )


# ---------------------------------------------------------------------------
# Empty-KB branch — every probe-4 key is present with value None
# ---------------------------------------------------------------------------


def test_markdown_backend_info_empty_kb_returns_all_keys_as_none(
    backend: MarkdownWikiBackend,
) -> None:
    """On a brand-new (no-pages) KB every probe-4 key is present and
    ``None``. Probe-4 forbids omission — markdown MUST surface ``None``.
    """
    info = backend.info(kb_id="probe4-empty")
    for key in _PROBE4_BACKEND_INFO_KEYS:
        assert key in info, (
            f"empty-KB markdown backend.info() omitted {key!r} — "
            "probe-4 contract requires the key to be present"
        )
        assert info[key] is None, (
            f"expected info()[{key!r}] is None on empty KB, got "
            f"{info[key]!r}"
        )

    assert info["backend"] == "markdown"
    assert info["spec_version"] == "2.1"


# ---------------------------------------------------------------------------
# Populated-KB branch — probe-4 fields populated, embedding stays None
# ---------------------------------------------------------------------------


def test_markdown_backend_info_populated_kb_surfaces_first_page_fields(
    backend: MarkdownWikiBackend,
) -> None:
    """After one ingest, probe-4 fields surface from the first page; the
    embedding-model field stays ``None`` (markdown does not embed).
    """
    kb_id = "probe4-pop"
    _ingest_one(backend, kb_id)
    info = backend.info(kb_id=kb_id)

    # Every probe-4 key still present.
    assert _PROBE4_BACKEND_INFO_KEYS.issubset(info), (
        f"populated-KB markdown backend.info() missing probe-4 keys: "
        f"{_PROBE4_BACKEND_INFO_KEYS - set(info)}"
    )

    # Populated fields reflect the first page.
    assert info["source_type"] == "file"
    assert info["uri"] == "/tmp/probe4.txt"
    assert info["dedup_key"] == "probe4-key"
    assert info["page_id"] is not None

    # CRITICAL: dominant_embedding_model is None, not omitted.
    assert info["dominant_embedding_model"] is None

    # Backend diagnostics are additive (probe-4 ignores them, but they
    # must not collide with the frozen keys).
    assert info["backend"] == "markdown"
    assert info["page_count"] == 1
    assert info["vector_count"] is None
    assert info["embedding_provider"] is None


# ---------------------------------------------------------------------------
# count() Protocol method
# ---------------------------------------------------------------------------


def test_markdown_backend_count_reflects_page_count(
    backend: MarkdownWikiBackend,
) -> None:
    """``count()`` returns the on-disk page count (markdown's analogue
    of ChromadbBackend.count's vector count)."""
    assert backend.count(kb_id="probe4-cnt") == 0
    _ingest_one(backend, "probe4-cnt")
    assert backend.count(kb_id="probe4-cnt") == 1


# ---------------------------------------------------------------------------
# health_check stays Protocol-compliant
# ---------------------------------------------------------------------------


def test_markdown_backend_health_check_returns_required_keys(
    backend: MarkdownWikiBackend,
) -> None:
    """``health_check()`` returns at minimum ``backend`` / ``status`` /
    ``spec_version`` per Protocol docstring."""
    health = backend.health_check()
    for key in ("backend", "status", "spec_version"):
        assert key in health, f"health_check missing required key {key!r}"
    assert health["backend"] == "markdown"
    assert health["spec_version"] == "2.1"
    assert health["status"] in {"ok", "degraded", "unavailable"}
