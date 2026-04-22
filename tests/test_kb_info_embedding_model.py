"""Acceptance tests: embedding model recorded per chunk; surfaced in kb_info.

Covers ROADMAP A4:
- Every ingested chunk carries ``metadata["embedding_model"]``.
- ``kb_info`` returns ``dominant_embedding_model`` (str | None) and
  ``embedding_model_counts`` (dict[str, int]).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.models import Chunk, SourceType
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService

_MODEL_NAME = "test-embedder"


@pytest.fixture()
def embedding_service(test_config: Settings) -> KnowledgebaseService:
    """KnowledgebaseService with a mocked embedder that exposes model_name."""
    svc = KnowledgebaseService(test_config)
    svc._embedder_instance = MagicMock()
    svc._embedder_instance.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    svc._embedder_instance.model_name = _MODEL_NAME
    svc._get_vectorstore = lambda kb_id: MagicMock(  # type: ignore[assignment]
        add=MagicMock(return_value=None),
        delete=MagicMock(return_value=None),
    )
    mock_ingestion = MagicMock()
    # Each ingest call returns two distinct chunks so we can assert a total count.
    mock_ingestion.ingest.side_effect = lambda *_, **__: [
        Chunk(source_id="", kb_id="", content="chunk one", metadata={}),
        Chunk(source_id="", kb_id="", content="chunk two", metadata={}),
    ]
    svc._ingestion = mock_ingestion
    return svc


# ---------------------------------------------------------------------------
# Test 1: dominant_embedding_model and embedding_model_counts when populated
# ---------------------------------------------------------------------------


def test_kb_info_returns_embedding_model_fields_when_populated(
    embedding_service: KnowledgebaseService,
) -> None:
    """After ingesting two sources (4 chunks total), kb_info exposes the model."""
    kb = embedding_service.create_kb("emb-model-info-kb")

    embedding_service.ingest_source(kb.id, SourceType.file, "/tmp/source_a.txt")
    embedding_service.ingest_source(kb.id, SourceType.file, "/tmp/source_b.txt")

    info = embedding_service.get_kb(kb.id)
    assert info is not None

    assert info.dominant_embedding_model == _MODEL_NAME
    assert info.embedding_model_counts == {_MODEL_NAME: 4}


# ---------------------------------------------------------------------------
# Test 2: fields are empty/None for a KB with no chunks
# ---------------------------------------------------------------------------


def test_kb_info_embedding_fields_empty_for_empty_kb(
    embedding_service: KnowledgebaseService,
) -> None:
    """A freshly created KB with zero chunks returns None / empty dict."""
    kb = embedding_service.create_kb("emb-model-empty-kb")

    info = embedding_service.get_kb(kb.id)
    assert info is not None

    assert info.dominant_embedding_model is None
    assert info.embedding_model_counts == {}


# ---------------------------------------------------------------------------
# Test 3: chunk rows in the DB carry the embedding_model metadata key
# ---------------------------------------------------------------------------


def test_chunk_metadata_records_embedding_model_on_ingest(
    embedding_service: KnowledgebaseService,
) -> None:
    """Every chunk stored in the DB must have metadata['embedding_model'] set."""
    kb = embedding_service.create_kb("emb-model-chunk-meta-kb")
    source = embedding_service.ingest_source(kb.id, SourceType.file, "/tmp/source_c.txt")

    # Retrieve chunks directly from the DB to verify persistence.
    ctx = embedding_service._ctx(kb.id)
    chunks = ctx.db.list_chunks(source.id)

    assert len(chunks) == 2, "Expected exactly 2 chunks from the mocked ingestion"
    for chunk in chunks:
        assert "embedding_model" in chunk.metadata, (
            f"chunk {chunk.id} missing 'embedding_model' in metadata: {chunk.metadata}"
        )
        assert chunk.metadata["embedding_model"] == _MODEL_NAME


# ---------------------------------------------------------------------------
# Test 4: chunk metadata also carries ``kb_id`` so the vectorstore's
# ``where={"kb_id": kb_id}`` filter returns results at query time.
# Regression — before this was added, kb_query returned zero results even
# when the per-KB collection held matching chunks.
# ---------------------------------------------------------------------------


def test_chunk_metadata_records_kb_id_for_query_filter(
    embedding_service: KnowledgebaseService,
) -> None:
    kb = embedding_service.create_kb("emb-model-kb-id-stamp")
    source = embedding_service.ingest_source(kb.id, SourceType.file, "/tmp/source_d.txt")

    ctx = embedding_service._ctx(kb.id)
    chunks = ctx.db.list_chunks(source.id)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.metadata.get("kb_id") == kb.id, (
            f"chunk {chunk.id} missing 'kb_id' in metadata: {chunk.metadata}"
        )
