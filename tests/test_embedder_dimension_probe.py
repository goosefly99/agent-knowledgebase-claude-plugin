"""Phase 0 Bug-1: ``Embedder.dimension`` no longer lies.

Three regressions pinned here, one per concrete embedder behaviour:

1. ``OpenAIEmbedder('qwen3-embedding:8b').dimension`` (the model that
   surfaced the bug — Ollama's ``/v1`` exposes it via the OpenAI-compatible
   surface, but it is not in the ``_REMOTE_EMBEDDING_DIMENSIONS`` lookup
   table, so the previous code returned the hardcoded 1536 fallback).
   With Phase 0 the embedder issues a single dry-run embed of the literal
   probe string and returns the response vector's length.
2. A model that IS in the lookup table (``text-embedding-3-small`` →
   1536) does NOT trigger an HTTP call — the static value is returned
   directly.
3. ``OllamaEmbedder.dimension`` triggers a probe on first access instead
   of returning ``0`` until the first regular embed completes.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import httpx
import pytest

from agent_knowledgebase.services.embeddings import (
    EmbedderUnavailableError,
    OllamaEmbedder,
    OpenAIEmbedder,
)


def _ok_remote_embed(vector: list[float]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "data": [{"index": 0, "embedding": vector}],
    }
    response.raise_for_status.return_value = None
    return response


def _ok_ollama_embed(vector: list[float]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "model": "qwen3-embedding:8b",
        "embeddings": [vector],
    }
    response.raise_for_status.return_value = None
    return response


def test_remote_embedder_unknown_model_probes_real_dimension() -> None:
    """Bug-1a: qwen3-embedding:8b is 4096-dim, not 1536.

    Pre-fix behaviour returned 1536 silently (the hardcoded fallback);
    post-fix behaviour probes the live endpoint and returns the actual
    vector length.
    """
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = _ok_remote_embed([0.0] * 4096)

    embedder = OpenAIEmbedder(
        model_name="qwen3-embedding:8b",
        api_key="test",
        base_url="http://test/v1",
        _client=mock_client,
    )

    assert embedder.dimension == 4096
    # The probe was issued against the embeddings endpoint with the
    # literal probe payload.
    mock_client.post.assert_called_once_with(
        "http://test/v1/embeddings",
        json={"input": ["probe"], "model": "qwen3-embedding:8b"},
    )

    # Subsequent accesses are cached — no further HTTP calls.
    assert embedder.dimension == 4096
    assert mock_client.post.call_count == 1


def test_remote_embedder_known_model_skips_probe() -> None:
    """Bug-1a fix preserves the static lookup for the 3 known OpenAI models.

    ``text-embedding-3-small`` is in ``_REMOTE_EMBEDDING_DIMENSIONS``
    with dim=1536; accessing ``.dimension`` must NOT issue an HTTP
    call.
    """
    mock_client = MagicMock(spec=httpx.Client)
    embedder = OpenAIEmbedder(
        model_name="text-embedding-3-small",
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        _client=mock_client,
    )

    assert embedder.dimension == 1536
    mock_client.post.assert_not_called()


def test_remote_embedder_known_model_probe_dimension_is_idempotent() -> None:
    """``probe_dimension`` for a known model returns the static value with no HTTP call."""
    mock_client = MagicMock(spec=httpx.Client)
    embedder = OpenAIEmbedder(
        model_name="text-embedding-3-large",
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        _client=mock_client,
    )

    assert embedder.probe_dimension() == 3072
    assert embedder.probe_dimension() == 3072
    mock_client.post.assert_not_called()


def test_ollama_embedder_dimension_triggers_probe() -> None:
    """Bug-1b: pre-fix `.dimension` returned 0 until first embed.

    The new behaviour issues a probe call on first access and caches
    the actual dimension reported by the response.
    """
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = _ok_ollama_embed([0.0] * 4096)

    embedder = OllamaEmbedder(
        model_name="qwen3-embedding:8b",
        base_url="http://127.0.0.1:11434",
        _client=mock_client,
    )

    # Sanity: cache starts unset (vs. the pre-fix `0` sentinel).
    assert embedder._dimension_cache is None

    # Accessing `.dimension` triggers a single probe.
    assert embedder.dimension == 4096
    mock_client.post.assert_called_once_with(
        "http://127.0.0.1:11434/api/embed",
        json={"model": "qwen3-embedding:8b", "input": ["probe"]},
    )

    # Subsequent accesses are cached.
    assert embedder.dimension == 4096
    assert mock_client.post.call_count == 1


def test_remote_probe_failure_reports_real_latency() -> None:
    """Important #1: empty-response probe must report measured latency.

    Pre-fix the empty-response branch hardcoded ``latency_ms=0``,
    masking slow-probe-then-empty failures. This test sleeps inside
    the mocked POST so any non-zero elapsed time proves the probe call
    site (not a literal ``0``) populated the field.
    """

    def slow_empty_response(*_args: object, **_kwargs: object) -> MagicMock:
        time.sleep(0.01)
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"data": []}
        response.raise_for_status.return_value = None
        return response

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = slow_empty_response

    embedder = OpenAIEmbedder(
        model_name="qwen3-embedding:8b",
        api_key="test",
        base_url="http://test/v1",
        _client=mock_client,
    )

    with pytest.raises(EmbedderUnavailableError) as excinfo:
        embedder.probe_dimension()

    assert excinfo.value.error == "probe_failed"
    assert excinfo.value.phase == "probe_dimension"
    assert excinfo.value.latency_ms > 0


def test_ollama_embedder_dimension_uses_cache_after_real_embed() -> None:
    """If a real embed already populated the cache, `.dimension` reuses it."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = _ok_ollama_embed([0.1, 0.2, 0.3])

    embedder = OllamaEmbedder(
        model_name="qwen3-embedding:8b",
        _client=mock_client,
    )

    # Real embed populates the cache via the existing `_call_embed`
    # path.
    embedder.embed_query("hello")
    mock_client.post.reset_mock()

    # `.dimension` must NOT issue another probe.
    assert embedder.dimension == 3
    mock_client.post.assert_not_called()
