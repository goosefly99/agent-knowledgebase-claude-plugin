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
    RemoteEmbedder,
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
# RemoteEmbedder — fully mocked via _client injection
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


class TestRemoteEmbedder:
    """Unit tests with a mocked httpx client injected via ``_client``."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        """Return a mock httpx.Client."""
        return MagicMock(spec=httpx.Client)

    @pytest.fixture()
    def embedder(self, mock_client: MagicMock) -> RemoteEmbedder:
        return RemoteEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test-key",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

    def test_dimension_text_embedding_3_small(self, embedder: RemoteEmbedder) -> None:
        assert embedder.dimension == 1536

    def test_dimension_text_embedding_3_large(self) -> None:
        emb = RemoteEmbedder(
            model_name="text-embedding-3-large",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=MagicMock(),
        )
        assert emb.dimension == 3072

    def test_dimension_text_embedding_ada_002(self) -> None:
        emb = RemoteEmbedder(
            model_name="text-embedding-ada-002",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=MagicMock(),
        )
        assert emb.dimension == 1536

    def test_dimension_unknown_model_defaults_1536(self) -> None:
        emb = RemoteEmbedder(
            model_name="some-future-model",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=MagicMock(),
        )
        assert emb.dimension == 1536

    def test_embed_calls_api(self, embedder: RemoteEmbedder) -> None:
        fake_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embedder._client.post.return_value = _make_embed_response(fake_vectors)

        result = embedder.embed(["text a", "text b"])

        embedder._client.post.assert_called_once_with(
            "https://example.invalid/v1/embeddings",
            json={"input": ["text a", "text b"], "model": "text-embedding-3-small"},
        )
        assert result == fake_vectors

    def test_embed_query_calls_api(self, embedder: RemoteEmbedder) -> None:
        fake_vectors = [[0.1, 0.2, 0.3]]
        embedder._client.post.return_value = _make_embed_response(fake_vectors)

        result = embedder.embed_query("hello")

        embedder._client.post.assert_called_once_with(
            "https://example.invalid/v1/embeddings",
            json={"input": ["hello"], "model": "text-embedding-3-small"},
        )
        assert result == [0.1, 0.2, 0.3]

    def test_embed_preserves_order_when_api_returns_unsorted(
        self, embedder: RemoteEmbedder
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
        emb = RemoteEmbedder(
            model_name="m",
            api_key="k",
            base_url="https://example.invalid/v1/",
            _client=mock_client,
        )
        emb.embed(["x"])
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "https://example.invalid/v1/embeddings"

    def test_satisfies_protocol(self, embedder: RemoteEmbedder) -> None:
        assert isinstance(embedder, Embedder)


# ---------------------------------------------------------------------------
# FIELD-14: bounded wall-clock + structured transport errors
# ---------------------------------------------------------------------------


class TestRemoteEmbedderTransportBounds:
    """RemoteEmbedder must bound its HTTP calls and surface structured errors."""

    def test_constructor_threads_timeout_and_auth_to_real_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _client is not injected, httpx.Client must receive the bounds + auth."""
        captured: dict[str, object] = {}

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(httpx, "Client", FakeClient)
        RemoteEmbedder(
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

    def test_empty_api_key_omits_auth_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty api_key must not send an Authorization header at all."""
        captured: dict[str, object] = {}

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(httpx, "Client", FakeClient)
        RemoteEmbedder(
            model_name="m",
            api_key="",
            base_url="http://localhost:11434/v1",
        )

        assert captured["headers"] == {}

    def test_embed_query_converts_timeout_to_unavailable_error(self) -> None:
        """httpx.TimeoutException must become EmbedderUnavailableError."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = httpx.ReadTimeout("read timed out")

        embedder = RemoteEmbedder(
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

        embedder = RemoteEmbedder(
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

        embedder = RemoteEmbedder(
            model_name="qwen3-embedding:8b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            _client=mock_client,
        )

        with pytest.raises(EmbedderUnavailableError) as excinfo:
            embedder.embed(["a", "b"])

        assert excinfo.value.to_payload()["phase"] == "embed_batch"

    def test_http_status_errors_pass_through_unchanged(self) -> None:
        """HTTP 4xx/5xx (auth failures etc.) must not be re-wrapped."""
        mock_client = MagicMock(spec=httpx.Client)
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=MagicMock(), response=MagicMock(status_code=401)
        )
        mock_client.post.return_value = response

        embedder = RemoteEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            _client=mock_client,
        )

        with pytest.raises(httpx.HTTPStatusError):
            embedder.embed_query("x")

    def test_unrelated_errors_pass_through_unchanged(self) -> None:
        """Non-transport errors (e.g. validation) must not be re-wrapped."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = ValueError("bad input")

        embedder = RemoteEmbedder(
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

        embedder = RemoteEmbedder(
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

        embedder = RemoteEmbedder(
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

    def test_creates_remote_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Factory returns RemoteEmbedder when provider='remote'."""
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="remote",
            embedding_model="text-embedding-3-small",
            embed_api_key="sk-test-key",
            embed_base_url="https://example.invalid/v1",
        )
        monkeypatch.setattr(httpx, "Client", MagicMock())
        embedder = create_embedder(config)
        assert isinstance(embedder, RemoteEmbedder)

    def test_remote_without_key_raises(self, tmp_path: Path) -> None:
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="remote",
            embedding_model="text-embedding-3-small",
            embed_api_key=None,
            embed_base_url="https://example.invalid/v1",
        )
        with pytest.raises(ValueError, match="AGENT_KB_EMBED_API_KEY required"):
            create_embedder(config)

    def test_remote_without_base_url_raises(self, tmp_path: Path) -> None:
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="remote",
            embedding_model="text-embedding-3-small",
            embed_api_key="sk-test",
            embed_base_url=None,
        )
        with pytest.raises(ValueError, match="AGENT_KB_EMBED_BASE_URL required"):
            create_embedder(config)

    def test_factory_threads_base_url_and_bounds_to_remote_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_embedder must pass base_url/timeout/max_retries from Settings."""
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="remote",
            embedding_model="qwen3-embedding:8b",
            embed_api_key="ollama",
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

        assert isinstance(embedder, RemoteEmbedder)
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

    def test_dimension_caches_after_first_call(self, embedder: OllamaEmbedder) -> None:
        assert embedder.dimension == 0  # no probe yet
        embedder._client.post.return_value = _make_ollama_response([[0.1] * 4096])
        embedder.embed_query("probe")
        assert embedder.dimension == 4096

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
        """The ollama provider must not fail when embed_api_key is None."""
        config = Settings(
            saves_dir=tmp_path,
            embedding_provider="ollama",
            embedding_model="qwen3-embedding:8b",
            embed_api_key=None,
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
