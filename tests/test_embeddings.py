"""Tests for agent_knowledgebase.services.embeddings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.services.embeddings import (
    DEFAULT_OLLAMA_BASE_URL,
    Embedder,
    EmbedderUnavailableError,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    create_embedder_for_model,
)


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder — real model, small & fast
# ---------------------------------------------------------------------------


class TestSentenceTransformerEmbedder:
    """Integration tests using the real all-MiniLM-L6-v2 model."""

    @pytest.fixture(scope="class")
    def embedder(self) -> SentenceTransformerEmbedder:
        """Load the model once per test class to avoid repeated downloads."""
        return SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    def test_dimension(self, embedder: SentenceTransformerEmbedder) -> None:
        assert embedder.dimension == 384

    def test_embed_returns_correct_count(self, embedder: SentenceTransformerEmbedder) -> None:
        texts = ["hello world", "foo bar baz"]
        result = embedder.embed(texts)
        assert len(result) == 2

    def test_embed_returns_correct_dimension(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed(["test sentence"])
        assert len(result[0]) == 384

    def test_embed_returns_float_lists(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed(["some text"])
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_query_returns_single_vector(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed_query("single query")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_empty_list(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed([])
        assert result == []

    def test_satisfies_protocol(self, embedder: SentenceTransformerEmbedder) -> None:
        assert isinstance(embedder, Embedder)


# ---------------------------------------------------------------------------
# OpenAIEmbedder — fully mocked via _client injection
# ---------------------------------------------------------------------------


def _make_embed_response(embeddings: list[list[float]], status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response carrying the given embedding payload."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = {
        "data": [
            {"embedding": emb, "index": idx} for idx, emb in enumerate(embeddings)
        ],
        "model": "test-model",
    }
    response.raise_for_status.return_value = None
    return response


class TestOpenAIEmbedder:
    """Unit tests with a mocked httpx client injected via ``_client``."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        """Return a mock httpx.Client."""
        return MagicMock(spec=httpx.Client)

    @pytest.fixture()
    def embedder(self, mock_client: MagicMock) -> OpenAIEmbedder:
        return OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test-key",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

    def test_dimension_text_embedding_3_small(self, embedder: OpenAIEmbedder) -> None:
        assert embedder.dimension == 1536

    def test_dimension_text_embedding_3_large(self) -> None:
        emb = OpenAIEmbedder(
            model_name="text-embedding-3-large",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=MagicMock(),
        )
        assert emb.dimension == 3072

    def test_dimension_text_embedding_ada_002(self) -> None:
        emb = OpenAIEmbedder(
            model_name="text-embedding-ada-002",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=MagicMock(),
        )
        assert emb.dimension == 1536

    def test_dimension_unknown_model_probes_via_dry_run(self) -> None:
        """Phase 0 Bug-1a fix: unknown model no longer hardcodes 1536.

        The previous behaviour returned the literal `1536` for any model
        not in `_REMOTE_EMBEDDING_DIMENSIONS`, silently producing a
        dimension mismatch on first ingest into a chromadb collection
        that was sized at the actual probed dimension. The new
        behaviour issues a dry-run embed of the literal probe string
        and returns the response vector's length.
        """
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = _make_embed_response([[0.0] * 4096])
        emb = OpenAIEmbedder(
            model_name="some-future-model",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )
        assert emb.dimension == 4096
        # The probe was a real HTTP call — verify it hit the embeddings
        # endpoint with the literal "probe" payload.
        mock_client.post.assert_called_once_with(
            "https://example.invalid/v1/embeddings",
            json={"input": ["probe"], "model": "some-future-model"},
        )

    def test_embed_calls_api(self, embedder: OpenAIEmbedder) -> None:
        fake_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embedder._client.post.return_value = _make_embed_response(fake_vectors)

        result = embedder.embed(["text a", "text b"])

        embedder._client.post.assert_called_once_with(
            "https://example.invalid/v1/embeddings",
            json={"input": ["text a", "text b"], "model": "text-embedding-3-small"},
        )
        assert result == fake_vectors

    def test_embed_query_calls_api(self, embedder: OpenAIEmbedder) -> None:
        fake_vectors = [[0.1, 0.2, 0.3]]
        embedder._client.post.return_value = _make_embed_response(fake_vectors)

        result = embedder.embed_query("hello")

        embedder._client.post.assert_called_once_with(
            "https://example.invalid/v1/embeddings",
            json={"input": ["hello"], "model": "text-embedding-3-small"},
        )
        assert result == [0.1, 0.2, 0.3]

    def test_embed_preserves_order_when_api_returns_unsorted(
        self, embedder: OpenAIEmbedder
    ) -> None:
        """Verify that results are sorted by index even if the API returns them out of order."""
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "data": [
                {"embedding": [0.4, 0.5], "index": 1},
                {"embedding": [0.1, 0.2], "index": 0},
            ]
        }
        response.raise_for_status.return_value = None
        embedder._client.post.return_value = response

        result = embedder.embed(["first", "second"])
        assert result == [[0.1, 0.2], [0.4, 0.5]]

    def test_base_url_trailing_slash_is_normalized(self) -> None:
        """A trailing slash on base_url must not produce '//embeddings'."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = _make_embed_response([[0.1]])
        emb = OpenAIEmbedder(
            model_name="m",
            api_key="k",
            base_url="https://example.invalid/v1/",
            _client=mock_client,
        )
        emb.embed(["x"])
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "https://example.invalid/v1/embeddings"

    def test_satisfies_protocol(self, embedder: OpenAIEmbedder) -> None:
        assert isinstance(embedder, Embedder)


# ---------------------------------------------------------------------------
# FIELD-14: bounded wall-clock + structured transport errors
# ---------------------------------------------------------------------------


class TestOpenAIEmbedderTransportBounds:
    """OpenAIEmbedder must bound its HTTP calls and surface structured errors."""

    def test_constructor_threads_timeout_and_auth_to_real_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _client is not injected, httpx.Client must receive the bounds + auth."""
        captured: dict[str, object] = {}

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(httpx, "Client", FakeClient)
        OpenAIEmbedder(
            model_name="qwen3-embedding:8b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout_seconds=12.5,
            max_retries=0,
        )

        assert captured["timeout"] == 12.5
        headers = captured["headers"]
        assert isinstance(headers, dict)
        assert headers["Authorization"] == "Bearer ollama"

    def test_empty_api_key_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty api_key must raise ValueError — OpenAI auth is required."""
        monkeypatch.setattr(httpx, "Client", MagicMock())
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbedder(
                model_name="m",
                api_key="",
                base_url="http://localhost:11434/v1",
            )

    def test_embed_query_converts_timeout_to_unavailable_error(self) -> None:
        """httpx.TimeoutException must become EmbedderUnavailableError."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.ReadTimeout("read timed out")

        embedder = OpenAIEmbedder(
            model_name="qwen3-embedding:8b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed_query("hello")

        payload = excinfo.value.to_payload()
        assert payload["error"] == "embed_timeout"
        assert payload["model"] == "qwen3-embedding:8b"
        assert payload["phase"] == "embed_query"
        assert isinstance(payload["latency_ms"], int)
        assert payload["latency_ms"] >= 0

    def test_embed_query_converts_connection_error_to_unavailable(self) -> None:
        """httpx.ConnectError (e.g. Ollama offline) must become EmbedderUnavailableError."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.ConnectError("connection refused")

        embedder = OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed_query("hello")

        assert excinfo.value.to_payload()["error"] == "embed_unreachable"

    def test_embed_batch_also_classifies_errors(self) -> None:
        """The batch path must also surface structured errors, not just embed_query."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.ReadTimeout("stalled")

        embedder = OpenAIEmbedder(
            model_name="qwen3-embedding:8b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed(["a", "b"])

        assert excinfo.value.to_payload()["phase"] == "embed_batch"

    def test_unclassified_http_status_errors_pass_through_unchanged(self) -> None:
        """Phase 0 Bug-1c: only known 4xx classes wrap; others pass through.

        Pre-Phase-0: every HTTP 4xx/5xx error propagated raw, including
        the auth-failure case which surfaces opaquely on the MCP layer.
        Post-Phase-0: the four known classes (404 model_not_pulled,
        401/403 auth_failed, 429 rate_limited) are wrapped as
        ``EmbedderUnavailableError``; everything else (e.g. 500) is
        still passed through so retry / circuit-breaker logic at higher
        layers stays informed.
        """
        mock_client = MagicMock(spec=httpx.Client)
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_client.post.return_value = response

        embedder = OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

        with pytest.raises(httpx.HTTPStatusError):
            embedder.embed_query("x")

    def test_http_401_classified_as_auth_failed(self) -> None:
        """Bug-1c: 401 → ``EmbedderUnavailableError(error='auth_failed')``."""
        mock_client = MagicMock(spec=httpx.Client)
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=MagicMock(status_code=401),
        )
        mock_client.post.return_value = response

        embedder = OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed_query("x")
        payload = excinfo.value.to_payload()
        assert payload["error"] == "auth_failed"
        assert payload["model"] == "text-embedding-3-small"
        assert payload["phase"] == "embed_query"

    def test_http_403_classified_as_auth_failed(self) -> None:
        """Bug-1c: 403 → ``EmbedderUnavailableError(error='auth_failed')``."""
        mock_client = MagicMock(spec=httpx.Client)
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Forbidden",
            request=MagicMock(),
            response=MagicMock(status_code=403),
        )
        mock_client.post.return_value = response

        embedder = OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed_query("x")
        assert excinfo.value.to_payload()["error"] == "auth_failed"

    def test_http_404_classified_as_model_not_pulled(self) -> None:
        """Bug-1c: 404 → ``EmbedderUnavailableError(error='model_not_pulled')``.

        Captures the Ollama-specific case where ``ollama pull
        <model>`` was never run on the host serving the
        ``/v1/embeddings`` surface.
        """
        mock_client = MagicMock(spec=httpx.Client)
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )
        mock_client.post.return_value = response

        embedder = OpenAIEmbedder(
            model_name="qwen3-embedding:8b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed_query("x")
        payload = excinfo.value.to_payload()
        assert payload["error"] == "model_not_pulled"
        assert payload["model"] == "qwen3-embedding:8b"

    def test_http_429_classified_as_rate_limited(self) -> None:
        """Bug-1c: 429 → ``EmbedderUnavailableError(error='rate_limited')``.

        Includes the ``Retry-After`` header value when the response
        carries one.
        """
        mock_client = MagicMock(spec=httpx.Client)
        mock_response = MagicMock(status_code=429)
        mock_response.headers = {"Retry-After": "30"}
        outer_response = MagicMock(spec=httpx.Response)
        outer_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=mock_response,
        )
        mock_client.post.return_value = outer_response

        embedder = OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed_query("x")
        payload = excinfo.value.to_payload()
        assert payload["error"] == "rate_limited"
        assert payload["retry_after"] == "30"

    def test_unrelated_errors_pass_through_unchanged(self) -> None:
        """Non-transport errors (e.g. validation) must not be re-wrapped."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = ValueError("bad input")

        embedder = OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

        with pytest.raises(ValueError, match="bad input"):
            embedder.embed_query("x")

    def test_retries_succeed_after_transient_failure(self) -> None:
        """max_retries=1 must retry once after a transport error, then succeed."""
        mock_client = MagicMock(spec=httpx.Client)
        good = _make_embed_response([[0.1, 0.2]])
        mock_client.post.side_effect = [
            httpx.ConnectError("first fails"),
            good,
        ]

        embedder = OpenAIEmbedder(
            model_name="m",
            api_key="k",
            base_url="http://localhost/v1",
            max_retries=1,
            _client=mock_client,
        )

        result = embedder.embed_query("x")
        assert result == [0.1, 0.2]
        assert mock_client.post.call_count == 2

    def test_retries_exhausted_raises_unavailable(self) -> None:
        """max_retries=1 with two failures must raise after 2 attempts."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = [
            httpx.ConnectError("first"),
            httpx.ConnectError("second"),
        ]

        embedder = OpenAIEmbedder(
            model_name="m",
            api_key="k",
            base_url="http://localhost/v1",
            max_retries=1,
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError):
            embedder.embed_query("x")

        assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# create_embedder factory
# ---------------------------------------------------------------------------


class TestCreateEmbedder:
    """Factory function tests."""

    def test_creates_sentence_transformer(self, tmp_path: Path) -> None:
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
        )
        embedder = create_embedder(config)
        assert isinstance(embedder, SentenceTransformerEmbedder)

    def test_creates_openai_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Factory returns OpenAIEmbedder when provider='openai'."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embed_base_url="https://example.invalid/v1",
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder(config)
        assert isinstance(embedder, OpenAIEmbedder)

    def test_openai_without_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embed_base_url="https://example.invalid/v1",
        )
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            create_embedder(config)

    def test_openai_without_base_url_uses_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When embed_base_url is unset, the OpenAI default URL is used."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embed_base_url=None,
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder(config)
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder._base_url == "https://api.openai.com/v1"

    def test_factory_threads_base_url_and_bounds_to_openai_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_embedder must pass base_url/timeout/max_retries from Settings."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embed_base_url="http://localhost:11434/v1",
            embed_timeout_seconds=5.0,
            embed_max_retries=0,
        )

        captured: dict[str, object] = {}

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(httpx, "Client", FakeClient)
        embedder = create_embedder(config)

        assert isinstance(embedder, OpenAIEmbedder)
        assert captured["timeout"] == 5.0
        # base_url is stored on the embedder, not the httpx.Client (we pass full URLs).
        assert embedder._base_url == "http://localhost:11434/v1"
        assert embedder._max_retries == 0


# ---------------------------------------------------------------------------
# OllamaEmbedder — native /api/embed endpoint
# ---------------------------------------------------------------------------


def _make_ollama_response(
    embeddings: list[list[float]], status_code: int = 200
) -> MagicMock:
    """Build a mock httpx.Response with Ollama's native embed shape."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = {
        "model": "qwen3-embedding:8b",
        "embeddings": embeddings,
    }
    response.raise_for_status.return_value = None
    return response


class TestOllamaEmbedder:
    """Unit tests for OllamaEmbedder with a mocked httpx client."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        return MagicMock(spec=httpx.Client)

    @pytest.fixture()
    def embedder(self, mock_client: MagicMock) -> OllamaEmbedder:
        return OllamaEmbedder(
            model_name="qwen3-embedding:8b",
            base_url="http://127.0.0.1:11434",
            _client=mock_client,
        )

    def test_embed_posts_to_api_embed(self, embedder: OllamaEmbedder) -> None:
        """Batch embed must POST to /api/embed with Ollama's input shape."""
        embedder._client.post.return_value = _make_ollama_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        result = embedder.embed(["a", "b"])

        embedder._client.post.assert_called_once_with(
            "http://127.0.0.1:11434/api/embed",
            json={"model": "qwen3-embedding:8b", "input": ["a", "b"]},
        )
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_query_returns_single_vector(self, embedder: OllamaEmbedder) -> None:
        embedder._client.post.return_value = _make_ollama_response([[0.5, 0.6, 0.7]])
        assert embedder.embed_query("hello") == [0.5, 0.6, 0.7]

    def test_embed_empty_list_returns_empty(self, embedder: OllamaEmbedder) -> None:
        """Empty input must short-circuit without hitting the wire."""
        assert embedder.embed([]) == []
        embedder._client.post.assert_not_called()

    def test_dimension_triggers_probe_when_uncached(
        self, embedder: OllamaEmbedder
    ) -> None:
        """Phase 0 Bug-1b fix: `.dimension` no longer returns 0 pre-embed.

        The previous behaviour returned 0 until a regular embed had
        populated the cache, which guaranteed a dimension-mismatch when
        a caller used `.dimension` to size a ChromaDB collection
        before any embed had run. The new behaviour issues a probe
        call on first `.dimension` access and caches the result.
        """
        embedder._client.post.return_value = _make_ollama_response(
            [[0.1] * 4096]
        )
        # First access triggers a probe.
        assert embedder.dimension == 4096
        # Probe sent the literal "probe" payload.
        embedder._client.post.assert_called_once_with(
            "http://127.0.0.1:11434/api/embed",
            json={"model": "qwen3-embedding:8b", "input": ["probe"]},
        )
        # Subsequent accesses are cached — no further HTTP calls.
        assert embedder.dimension == 4096
        assert embedder._client.post.call_count == 1

    def test_timeout_converts_to_unavailable(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.ReadTimeout("stalled")
        embedder = OllamaEmbedder(
            model_name="qwen3-embedding:8b", _client=mock_client
        )
        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed_query("x")
        payload = excinfo.value.to_payload()
        assert payload["error"] == "embed_timeout"
        assert payload["phase"] == "embed_query"

    def test_connection_error_converts_to_unavailable(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.ConnectError("refused")
        embedder = OllamaEmbedder(
            model_name="qwen3-embedding:8b", _client=mock_client
        )
        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed_query("x")
        assert excinfo.value.to_payload()["error"] == "embed_unreachable"

    def test_trailing_slash_on_base_url_normalized(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = _make_ollama_response([[0.1]])
        embedder = OllamaEmbedder(
            model_name="m",
            base_url="http://127.0.0.1:11434/",
            _client=mock_client,
        )
        embedder.embed(["x"])
        assert (
            mock_client.post.call_args[0][0]
            == "http://127.0.0.1:11434/api/embed"
        )

    def test_satisfies_protocol(self, embedder: OllamaEmbedder) -> None:
        assert isinstance(embedder, Embedder)


class TestCreateEmbedderOllama:
    """Factory tests for the 'ollama' provider."""

    def test_creates_ollama_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="ollama",
            embedding_model="qwen3-embedding:8b",
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder(config)
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder._base_url == DEFAULT_OLLAMA_BASE_URL

    def test_ollama_uses_configured_base_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="ollama",
            embedding_model="qwen3-embedding:8b",
            embed_base_url="http://10.0.0.5:11434",
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder(config)
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder._base_url == "http://10.0.0.5:11434"

    def test_ollama_does_not_require_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ollama provider must not require any API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="ollama",
            embedding_model="qwen3-embedding:8b",
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder(config)
        assert isinstance(embedder, OllamaEmbedder)


class TestCreateEmbedderForModel:
    """Factory tests for create_embedder_for_model (per-KB query builder)."""

    def test_returns_default_when_model_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="ollama",
            embedding_model="qwen3-embedding:8b",
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder_for_model(config, "qwen3-embedding:8b")
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.model_name == "qwen3-embedding:8b"

    def test_returns_default_when_model_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="ollama",
            embedding_model="qwen3-embedding:8b",
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder_for_model(config, None)
        assert embedder.model_name == "qwen3-embedding:8b"

    def test_overrides_model_for_different_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the stored model differs from config, build an embedder
        for the stored model — otherwise retrieval silently breaks."""
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="ollama",
            embedding_model="qwen3-embedding:8b",
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder_for_model(config, "bge-m3:latest")
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.model_name == "bge-m3:latest"
