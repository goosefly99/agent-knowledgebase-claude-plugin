"""Phase 0 Bug-1a: dimension-mismatch is no longer silent at the embedder layer.

PHASE 0 SCOPE NOTE
==================

The full ``ChromaDBStore.expected_dim`` plumbing — passing ``expected_dim``
into the constructor, asserting against ``collection.metadata['dim']``, and
setting ``dim`` on collection creation — is **deferred to Phase 2**. That
work changes ``services/vectorstore.py`` ``ChromaDBStore.__init__`` and
the ``KnowledgebaseService`` construction path that calls
``create_vectorstore``, neither of which falls inside Phase 0's
"no architecture change" boundary. Both the spec brief
(``pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json``
``implementation.phases[0]``) and the subagent task description
explicitly authorise this deferral.

What this file pins instead is the **root** of Bug-1a as fixed in
Phase 0:

* Pre-fix: ``RemoteEmbedder('qwen3-embedding:8b').dimension`` silently
  returned the hardcoded ``1536`` fallback. Any caller that sized a
  ChromaDB collection from that value, then issued the first real embed
  (which produced 4096-dim vectors), would dimension-mismatch on
  insert. The bug was at the EMBEDDER layer, not the chromadb layer.
* Post-fix: ``RemoteEmbedder.dimension`` probes the live endpoint for
  unknown models and caches the actual response length. The mismatch
  source is removed.

The new ``EmbedderDimensionMismatchError`` exception is added to
``services/embeddings.py`` so the Phase 2 ``ChromaDBStore.expected_dim``
guard has a structured exception to raise. Phase 2 will add a
``test_chromadb_collection_dim_metadata_enforced`` test against that
plumbing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from agent_knowledgebase.services.embeddings import (
    EmbedderDimensionMismatchError,
    RemoteEmbedder,
)


def _ok_remote_embed(vector: list[float]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "data": [{"index": 0, "embedding": vector}],
    }
    response.raise_for_status.return_value = None
    return response


def test_qwen3_embedding_no_longer_silently_returns_1536() -> None:
    """Bug-1a regression: the hardcoded fallback is gone.

    The exact pathology the spec calls out:
    ``RemoteEmbedder('qwen3-embedding:8b').dimension`` previously
    returned ``1536`` without any HTTP call (because the model name is
    not in ``_REMOTE_EMBEDDING_DIMENSIONS``). Post-fix it probes the
    live endpoint and returns the response vector's length — proving
    the silent-mismatch source is gone.
    """
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = _ok_remote_embed([0.0] * 4096)

    embedder = RemoteEmbedder(
        model_name="qwen3-embedding:8b",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        _client=mock_client,
    )

    dim = embedder.dimension
    assert dim != 1536, (
        "RemoteEmbedder.dimension must NOT silently return the hardcoded "
        "1536 fallback for unknown models; this was the root cause of "
        "Bug-1a's dimension-mismatch on first ingest."
    )
    assert dim == 4096


def test_embedder_dimension_mismatch_error_payload_shape() -> None:
    """The new exception carries enough metadata for a structured response.

    Phase 2 will plumb this through ``ChromaDBStore.expected_dim`` so
    that querying a 1536-dim collection with a 4096-dim embedder
    raises ``EmbedderDimensionMismatchError`` instead of the generic
    chromadb ``InvalidDimensionException``.
    """
    err = EmbedderDimensionMismatchError(
        expected=384,
        actual=1536,
        model="qwen3-embedding:8b",
        collection="kb_abc123",
    )

    payload = err.to_payload()
    assert payload["error"] == "dimension_mismatch"
    assert payload["expected"] == 384
    assert payload["actual"] == 1536
    assert payload["model"] == "qwen3-embedding:8b"
    assert payload["collection"] == "kb_abc123"


def test_embedder_dimension_mismatch_error_is_a_runtime_error() -> None:
    """Code that already catches ``RuntimeError`` keeps working."""
    err = EmbedderDimensionMismatchError(expected=384, actual=1536)
    assert isinstance(err, RuntimeError)
    with pytest.raises(EmbedderDimensionMismatchError):
        raise err
