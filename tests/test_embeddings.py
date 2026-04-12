"""Tests for agent_knowledgebase.services.embeddings."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.services.embeddings import (
    Embedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
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

    def test_embed_returns_correct_dimension(
        self, embedder: SentenceTransformerEmbedder
    ) -> None:
        result = embedder.embed(["test sentence"])
        assert len(result[0]) == 384

    def test_embed_returns_float_lists(self, embedder: SentenceTransformerEmbedder) -> None:
        result = embedder.embed(["some text"])
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_query_returns_single_vector(
        self, embedder: SentenceTransformerEmbedder
    ) -> None:
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


def _make_openai_response(embeddings: list[list[float]]) -> SimpleNamespace:
    """Build a mock OpenAI embeddings response."""
    data = []
    for idx, emb in enumerate(embeddings):
        item = SimpleNamespace(embedding=emb, index=idx)
        data.append(item)
    return SimpleNamespace(data=data)


class TestOpenAIEmbedder:
    """Unit tests with a mocked OpenAI client injected via ``_client``."""

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        """Return a mock OpenAI client."""
        return MagicMock()

    @pytest.fixture()
    def embedder(self, mock_client: MagicMock) -> OpenAIEmbedder:
        """Return an OpenAIEmbedder with a mocked client."""
        return OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test-key",
            _client=mock_client,
        )

    def test_dimension_text_embedding_3_small(self, embedder: OpenAIEmbedder) -> None:
        assert embedder.dimension == 1536

    def test_dimension_text_embedding_3_large(self) -> None:
        emb = OpenAIEmbedder(
            model_name="text-embedding-3-large", api_key="sk-test", _client=MagicMock()
        )
        assert emb.dimension == 3072

    def test_dimension_text_embedding_ada_002(self) -> None:
        emb = OpenAIEmbedder(
            model_name="text-embedding-ada-002", api_key="sk-test", _client=MagicMock()
        )
        assert emb.dimension == 1536

    def test_dimension_unknown_model_defaults_1536(self) -> None:
        emb = OpenAIEmbedder(
            model_name="some-future-model", api_key="sk-test", _client=MagicMock()
        )
        assert emb.dimension == 1536

    def test_embed_calls_api(self, embedder: OpenAIEmbedder) -> None:
        fake_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embedder._client.embeddings.create.return_value = _make_openai_response(fake_vectors)

        result = embedder.embed(["text a", "text b"])

        embedder._client.embeddings.create.assert_called_once_with(
            input=["text a", "text b"], model="text-embedding-3-small"
        )
        assert result == fake_vectors

    def test_embed_query_calls_api(self, embedder: OpenAIEmbedder) -> None:
        fake_vectors = [[0.1, 0.2, 0.3]]
        embedder._client.embeddings.create.return_value = _make_openai_response(fake_vectors)

        result = embedder.embed_query("hello")

        embedder._client.embeddings.create.assert_called_once_with(
            input=["hello"], model="text-embedding-3-small"
        )
        assert result == [0.1, 0.2, 0.3]

    def test_embed_preserves_order_when_api_returns_unsorted(
        self, embedder: OpenAIEmbedder
    ) -> None:
        """Verify that results are sorted by index even if the API returns them out of order."""
        data = [
            SimpleNamespace(embedding=[0.4, 0.5], index=1),
            SimpleNamespace(embedding=[0.1, 0.2], index=0),
        ]
        embedder._client.embeddings.create.return_value = SimpleNamespace(data=data)

        result = embedder.embed(["first", "second"])
        assert result == [[0.1, 0.2], [0.4, 0.5]]

    def test_satisfies_protocol(self, embedder: OpenAIEmbedder) -> None:
        assert isinstance(embedder, Embedder)


# ---------------------------------------------------------------------------
# create_embedder factory
# ---------------------------------------------------------------------------


class TestCreateEmbedder:
    """Factory function tests."""

    def test_creates_sentence_transformer(self) -> None:
        config = Settings(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
        )
        embedder = create_embedder(config)
        assert isinstance(embedder, SentenceTransformerEmbedder)

    def test_creates_openai_embedder(self) -> None:
        """Factory returns OpenAIEmbedder; real OpenAI import is avoided via _client."""
        config = Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key="sk-test-key",
        )
        # create_embedder calls OpenAIEmbedder() which tries to import openai.
        # Since openai is an optional dep, we monkeypatch the class to inject a mock client.
        original_init = OpenAIEmbedder.__init__

        def patched_init(self: OpenAIEmbedder, **kwargs: object) -> None:
            kwargs["_client"] = MagicMock()
            original_init(self, **kwargs)  # type: ignore[arg-type]

        OpenAIEmbedder.__init__ = patched_init  # type: ignore[assignment]
        try:
            embedder = create_embedder(config)
        finally:
            OpenAIEmbedder.__init__ = original_init  # type: ignore[assignment]

        assert isinstance(embedder, OpenAIEmbedder)

    def test_openai_without_key_raises(self) -> None:
        config = Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key=None,
        )
        with pytest.raises(ValueError, match="AGENT_KB_OPENAI_API_KEY required"):
            create_embedder(config)
