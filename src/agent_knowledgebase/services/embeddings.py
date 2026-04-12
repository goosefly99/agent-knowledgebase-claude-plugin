"""Embedding providers behind a common interface.

Supports SentenceTransformers (default, local) and OpenAI (remote, requires API key).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_knowledgebase.config import Settings


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Common interface that all embedding providers must satisfy."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of texts, returning one vector per text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


# ---------------------------------------------------------------------------
# SentenceTransformers provider
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Local embedder backed by the ``sentence-transformers`` library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed *texts* and return a list of float vectors."""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension reported by the loaded model."""
        return int(self._model.get_embedding_dimension())


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

# Known dimensions for OpenAI embedding models.
_OPENAI_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """Remote embedder backed by the OpenAI embeddings API."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        _client: Any | None = None,
    ) -> None:
        self._model_name = model_name
        if _client is not None:
            self._client = _client
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed *texts* via the OpenAI API."""
        response = self._client.embeddings.create(input=texts, model=self._model_name)
        # The API returns data sorted by index; sort explicitly for safety.
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension for the configured model.

        Falls back to 1536 for unknown model names.
        """
        return _OPENAI_DIMENSIONS.get(self._model_name, 1536)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_embedder(config: Settings) -> Embedder:
    """Instantiate the appropriate :class:`Embedder` based on *config*.

    Raises:
        ValueError: If the OpenAI provider is selected but no API key is set.
    """
    if config.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=config.embedding_model)
    elif config.embedding_provider == "openai":
        if not config.openai_api_key:
            raise ValueError("AGENT_KB_OPENAI_API_KEY required for OpenAI embeddings")
        return OpenAIEmbedder(
            model_name=config.embedding_model,
            api_key=config.openai_api_key,
        )
    else:
        # Should be unreachable due to the Literal type constraint, but guard anyway.
        raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")
