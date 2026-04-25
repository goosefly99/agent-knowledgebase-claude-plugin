"""Phase 2 contract test: probe-4 response shape preserved across backends.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        constraints[4] (probe-4 minimum response shape frozen)
        + risks[0] mitigation (validation finding f-20)

The data-etl-orchestrator >= v0.4.0 plugin's probe-4 introspection step
inspects ``kb_info``, ``kb_list_pages``, and ``kb_list_sources`` for
five frozen keys: ``source_type``, ``uri``, ``dedup_key``, ``page_id``,
``dominant_embedding_model``. New backends (markdown, lightrag) MUST
return ``None`` for inapplicable fields — they MUST NOT omit them. This
test pins that contract for the chromadb backend (Phase 2 default);
``tests/contract/test_markdown_backend_probe4_shape.py`` (Phase 3 deliverable)
will pin the same contract for the markdown backend.

COORDINATION NOTE
-----------------

data-etl-orchestrator >= v0.4.0 must accept
``dominant_embedding_model = null`` before Phase 3 ships the markdown
backend (which has no concept of embeddings). The chromadb backend
populates the field whenever the KB has at least one ingested chunk,
so this test exercises both branches: empty KB -> ``None``; populated
KB -> a real model name.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.models import Chunk, SourceType
from agent_knowledgebase.server import (
    kb_info,
    kb_list_pages,
    kb_list_sources,
)
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# ---------------------------------------------------------------------------
# Frozen probe-4 keys per spec.constraints[4]
# ---------------------------------------------------------------------------

# Probe-4 inspects FIVE keys total, distributed across the three frozen
# read-path tools. Each tool surfaces the subset of fields it owns:
#
#   kb_info        -> dominant_embedding_model (KB-scoped aggregate)
#   kb_list_pages  -> page_id + first-source fields (source_type, uri, dedup_key)
#   kb_list_sources-> source_id + per-source fields (source_type, uri, dedup_key)
#
# The full union {source_type, uri, dedup_key, page_id,
# dominant_embedding_model} is the probe-4 frozen set per
# spec.constraints[4]. Backends MUST surface the union via these three
# tools and MUST return null for inapplicable fields (NOT omit them) —
# the markdown backend (Phase 3) returns dominant_embedding_model=null.

_PROBE4_KB_INFO_KEYS = {"dominant_embedding_model"}
_PROBE4_PAGE_KEYS = {"source_type", "uri", "dedup_key", "page_id"}
_PROBE4_SOURCE_KEYS = {"source_type", "uri", "dedup_key", "source_id"}

# Probe-4's union view — the full frozen set spanning the three tools.
_PROBE4_UNION = (
    _PROBE4_KB_INFO_KEYS
    | _PROBE4_PAGE_KEYS
    | _PROBE4_SOURCE_KEYS - {"source_id"}
    | {"page_id"}
)
# The five-key union per spec.constraints[4]:
assert _PROBE4_UNION == {
    "source_type",
    "uri",
    "dedup_key",
    "page_id",
    "dominant_embedding_model",
}, "probe-4 union drift — see spec.constraints[4]"

# Backend.info() (the new RetrieverBackend method) returns the FULL
# union since it is backend-side, not MCP-side. Each backend's info()
# is the single point that must surface every probe-4 key directly.
_PROBE4_BACKEND_INFO_KEYS = _PROBE4_UNION


_FIXTURE_MODEL = "probe4-test-embedder"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def chromadb_service(test_config: Settings) -> KnowledgebaseService:
    """Build a chromadb-backed service with mocked embedder/vectorstore.

    The embedder mock stamps a known ``model_name`` so we can assert
    ``dominant_embedding_model`` round-trips through ``kb_info``.
    """
    svc = KnowledgebaseService(test_config)
    svc._embedder_instance = MagicMock()
    svc._embedder_instance.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    svc._embedder_instance.model_name = _FIXTURE_MODEL
    svc._get_vectorstore = lambda kb_id: MagicMock(  # type: ignore[assignment]
        add=MagicMock(return_value=None),
        delete=MagicMock(return_value=None),
        count=MagicMock(return_value=0),
    )
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.side_effect = lambda *_, **__: [
        Chunk(source_id="", kb_id="", content="probe4 fixture chunk", metadata={})
    ]
    svc._ingestion = mock_ingestion
    return svc


# ---------------------------------------------------------------------------
# kb_info shape
# ---------------------------------------------------------------------------


def test_kb_info_returns_dominant_embedding_model_for_populated_kb(
    chromadb_service: KnowledgebaseService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ingest, ``kb_info`` JSON exposes a non-null
    ``dominant_embedding_model`` (the probe-4 field this tool owns).
    """
    monkeypatch.setattr("agent_knowledgebase.server._get_service", lambda: chromadb_service)

    kb = chromadb_service.create_kb("probe4-populated")
    chromadb_service.ingest_source(
        kb.id,
        SourceType.file,
        "/tmp/probe4_doc.txt",
        dedup_key="probe4-key",
    )

    payload = json.loads(kb_info(kb.id))

    assert _PROBE4_KB_INFO_KEYS.issubset(payload), (
        f"kb_info missing probe-4 keys: "
        f"{_PROBE4_KB_INFO_KEYS - set(payload)}"
    )
    assert payload["dominant_embedding_model"] == _FIXTURE_MODEL


def test_kb_info_dominant_embedding_model_is_null_not_omitted_for_empty_kb(
    chromadb_service: KnowledgebaseService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``kb_info`` on an empty KB returns
    ``dominant_embedding_model = null`` (NOT omitted).

    This branch is the structural pre-condition for Phase 3's markdown
    backend, which always returns null since markdown does not embed.
    """
    monkeypatch.setattr("agent_knowledgebase.server._get_service", lambda: chromadb_service)

    kb = chromadb_service.create_kb("probe4-empty")
    payload = json.loads(kb_info(kb.id))

    assert "dominant_embedding_model" in payload, (
        "kb_info omitted dominant_embedding_model on empty KB — probe-4 "
        "requires the field to be present with value null"
    )
    assert payload["dominant_embedding_model"] is None


# ---------------------------------------------------------------------------
# kb_list_pages shape
# ---------------------------------------------------------------------------


def test_kb_list_pages_each_entry_has_probe4_keys(
    chromadb_service: KnowledgebaseService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every page in ``kb_list_pages`` carries the probe-4 first-source
    fields (``source_type``, ``uri``, ``dedup_key``) and the
    ``page_id`` alias.
    """
    monkeypatch.setattr("agent_knowledgebase.server._get_service", lambda: chromadb_service)

    kb = chromadb_service.create_kb("probe4-pages")
    chromadb_service.ingest_source(
        kb.id,
        SourceType.file,
        "/tmp/probe4_pages.txt",
        dedup_key="probe4-pages-key",
    )

    pages = json.loads(kb_list_pages(kb.id))
    assert pages, "expected at least one page after ingest"
    for page in pages:
        missing = _PROBE4_PAGE_KEYS - set(page)
        assert not missing, (
            f"kb_list_pages entry missing probe-4 keys: {sorted(missing)} "
            f"in {page!r}"
        )


# ---------------------------------------------------------------------------
# kb_list_sources shape
# ---------------------------------------------------------------------------


def test_kb_list_sources_each_entry_has_probe4_keys(
    chromadb_service: KnowledgebaseService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every source in ``kb_list_sources`` carries probe-4 keys
    (``source_type``, ``uri``, ``dedup_key``, ``source_id``).
    """
    monkeypatch.setattr("agent_knowledgebase.server._get_service", lambda: chromadb_service)

    kb = chromadb_service.create_kb("probe4-sources")
    chromadb_service.ingest_source(
        kb.id,
        SourceType.website,
        "https://example.com/probe4",
        dedup_key="probe4-sources-key",
    )

    sources = json.loads(kb_list_sources(kb.id))
    assert sources, "expected at least one source after ingest"
    for source in sources:
        missing = _PROBE4_SOURCE_KEYS - set(source)
        assert not missing, (
            f"kb_list_sources entry missing probe-4 keys: {sorted(missing)} "
            f"in {source!r}"
        )


# ---------------------------------------------------------------------------
# Backend.info() direct probe (independent of MCP serialization)
# ---------------------------------------------------------------------------


def test_chromadb_backend_info_returns_full_probe4_union_directly(
    chromadb_service: KnowledgebaseService,
) -> None:
    """``ChromadbBackend.info(kb_id=...)`` itself returns the FULL
    probe-4 union (5 keys) — backend-side guard independent of the
    MCP / Pydantic serialization layer.

    The MCP-side guard splits the union across kb_info /
    kb_list_pages / kb_list_sources. Markdown / lightrag backends in
    later phases will reuse this exact test against their own
    ``info()`` implementations.
    """
    kb = chromadb_service.create_kb("probe4-backend-direct")
    chromadb_service.ingest_source(
        kb.id,
        SourceType.file,
        "/tmp/probe4_backend.txt",
        dedup_key="probe4-backend-key",
    )

    info = chromadb_service._backend.info(kb_id=kb.id)
    assert _PROBE4_BACKEND_INFO_KEYS.issubset(info), (
        f"backend.info() missing probe-4 keys: "
        f"{_PROBE4_BACKEND_INFO_KEYS - set(info)}"
    )
    assert info["source_type"] == "file"
    assert info["uri"] == "/tmp/probe4_backend.txt"
    assert info["dedup_key"] == "probe4-backend-key"
    assert info["dominant_embedding_model"] == _FIXTURE_MODEL
    assert info["page_id"] is not None  # the auto-created summary page


def test_chromadb_backend_info_uses_none_not_omission_for_empty_kb(
    chromadb_service: KnowledgebaseService,
) -> None:
    """Probe-4 contract: inapplicable fields are ``None``, never absent."""
    kb = chromadb_service.create_kb("probe4-empty-backend")
    info = chromadb_service._backend.info(kb_id=kb.id)
    for key in _PROBE4_BACKEND_INFO_KEYS:
        assert key in info, f"empty-KB backend.info() omitted {key!r}"
        # All probe-4 fields are None on empty KB (no sources, no chunks).
        assert info[key] is None, (
            f"expected backend.info()[{key!r}] == None on empty KB, "
            f"got {info[key]!r}"
        )
