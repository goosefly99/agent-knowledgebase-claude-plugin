"""Acceptance tests: kb_list_pages and kb_list_sources additive fields."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.models import Chunk, SourceType
from agent_knowledgebase.server import kb_list_pages, kb_list_sources
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


@pytest.fixture()
def augment_service(test_config: Settings) -> KnowledgebaseService:
    """KnowledgebaseService with mocked embedder, vectorstore, and ingestion."""
    svc = KnowledgebaseService(test_config)
    svc._embedder_instance = MagicMock()
    svc._embedder_instance.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    svc._get_vectorstore = lambda kb_id: MagicMock(  # type: ignore[assignment]
        add=MagicMock(return_value=None),
        delete=MagicMock(return_value=None),
    )
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.side_effect = lambda *_, **__: [
        Chunk(source_id="", kb_id="", content="hello world", metadata={})
    ]
    svc._ingestion = mock_ingestion
    return svc


def test_kb_list_pages_includes_source_fields(
    augment_service: KnowledgebaseService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kb_list_pages response includes source_type, uri, dedup_key, page_id per page."""
    monkeypatch.setattr("agent_knowledgebase.server._get_service", lambda: augment_service)

    kb = augment_service.create_kb("augment-pages-kb")
    augment_service.ingest_source(
        kb.id,
        SourceType.file,
        "/tmp/test_doc.txt",
        dedup_key="doc-key-1",
    )

    result_json = kb_list_pages(kb.id)
    pages = json.loads(result_json)

    assert len(pages) >= 1
    page = pages[0]

    # New additive fields present.
    assert "source_type" in page
    assert "uri" in page
    assert "dedup_key" in page
    assert "page_id" in page

    # Values are correct.
    assert page["source_type"] == "file"
    assert page["uri"] == "/tmp/test_doc.txt"
    assert page["dedup_key"] == "doc-key-1"
    assert page["page_id"] == page["id"]

    # Existing fields still present and untouched.
    for field in ("id", "kb_id", "title", "content", "page_type", "source_ids",
                  "tags", "created_at", "updated_at", "inbound_links", "outbound_links"):
        assert field in page, f"Existing field '{field}' missing from response"


def test_kb_list_sources_includes_source_id(
    augment_service: KnowledgebaseService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kb_list_sources response includes source_id equal to id per source."""
    monkeypatch.setattr("agent_knowledgebase.server._get_service", lambda: augment_service)

    kb = augment_service.create_kb("augment-sources-kb")
    augment_service.ingest_source(
        kb.id,
        SourceType.website,
        "https://example.com/docs",
        dedup_key="web-key-1",
    )

    result_json = kb_list_sources(kb.id)
    sources = json.loads(result_json)

    assert len(sources) >= 1
    source = sources[0]

    # New additive field present and aliased correctly.
    assert "source_id" in source
    assert source["source_id"] == source["id"]

    # Existing fields still present and untouched.
    for field in ("id", "kb_id", "source_type", "uri", "metadata",
                  "ingested_at", "chunk_count", "status", "dedup_key"):
        assert field in source, f"Existing field '{field}' missing from response"
