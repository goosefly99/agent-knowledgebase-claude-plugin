"""Embedding providers behind a common interface.

Supported providers:

* ``ollama`` (default) — POSTs to Ollama's native ``/api/embed`` endpoint.
  Requires only ``base_url`` (default ``http://127.0.0.1:11434``); no API
  key. Parses the Ollama response shape ``{"embeddings": [[...]]}``.
* ``remote`` — POSTs to an HTTP ``/v1/embeddings`` endpoint speaking the
  OpenAI-compatible embeddings JSON contract (``{"input": [...], "model":
  ...}`` → ``{"data": [{"index": N, "embedding": [...]}]}``). Works with
  vLLM, LocalAI, OpenAI, Azure OpenAI, Ollama's ``/v1`` surface, etc.
* ``sentence-transformers`` — Local embedder backed by the
  ``sentence-transformers`` library (offline-capable).
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import httpx

from agent_knowledgebase.config import Settings


# ---------------------------------------------------------------------------
# Structured error for unreachable / timed-out remote embedders
# ---------------------------------------------------------------------------


class EmbedderUnavailableError(RuntimeError):
    """Remote embedder backend is unreachable or exceeded its wall-clock bound.

    Carries a FIELD-14 structured payload so MCP tools can return a JSON
    error to the caller instead of blocking the RPC until the transport's
    own default timeout eventually fires.

    Carries an optional ``extra`` mapping for HTTP-status classification
    fields (e.g. ``retry_after`` on 429 responses) that don't fit the
    fixed-shape FIELD-14 payload.
    """

    def __init__(
        self,
        *,
        error: str,
        model: str,
        phase: str,
        latency_ms: int,
        detail: str = "",
        extra: dict[str, object] | None = None,
    ) -> None:
        self.error = error
        self.model = model
        self.phase = phase
        self.latency_ms = latency_ms
        self.detail = detail
        self.extra = dict(extra) if extra else {}
        super().__init__(
            f"{error}: model={model} phase={phase} latency_ms={latency_ms}"
        )

    def to_payload(self) -> dict[str, object]:
        """Return the FIELD-14 JSON-serialisable payload shape."""
        payload: dict[str, object] = {
            "error": self.error,
            "model": self.model,
            "phase": self.phase,
            "latency_ms": self.latency_ms,
        }
        if self.detail:
            payload["detail"] = self.detail
        for key, value in self.extra.items():
            payload[key] = value
        return payload


class EmbedderDimensionMismatchError(RuntimeError):
    """An embedder's vector dimension does not match the collection's expected dim.

    Raised when an attempt is made to ingest into or query a vector
    collection with vectors whose dimensionality differs from the
    dimension the collection was originally created with. Carries the
    expected and actual dimensions so callers can return a structured
    error payload.
    """

    def __init__(
        self,
        *,
        expected: int,
        actual: int,
        model: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.model = model
        self.collection = collection
        msg = (
            f"embedder dimension mismatch: expected={expected} actual={actual}"
        )
        if model:
            msg += f" model={model}"
        if collection:
            msg += f" collection={collection}"
        super().__init__(msg)

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serialisable payload shape."""
        payload: dict[str, object] = {
            "error": "dimension_mismatch",
            "expected": self.expected,
            "actual": self.actual,
        }
        if self.model is not None:
            payload["model"] = self.model
        if self.collection is not None:
            payload["collection"] = self.collection
        return payload


def _classify_httpx_transport_error(exc: BaseException) -> str | None:
    """Return a stable error token for known httpx transport errors.

    ``None`` means the exception is not a bounded-embedder transport error
    and should propagate unchanged (e.g. HTTP 4xx/5xx from the server,
    validation errors, etc.).
    """
    if isinstance(exc, httpx.TimeoutException):
        return "embed_timeout"
    if isinstance(exc, httpx.NetworkError):
        return "embed_unreachable"
    return None


def _classify_http_status_error(
    exc: httpx.HTTPStatusError,
) -> tuple[str, dict[str, object]] | None:
    """Return ``(error_token, extra_fields)`` for known HTTP 4xx classes.

    Returns ``None`` when the status is not a class we want to wrap as an
    :class:`EmbedderUnavailableError` (e.g. unexpected 5xx — let those
    propagate raw so retry logic at higher layers stays informed).
    """
    response = exc.response
    if response is None:
        return None
    status = getattr(response, "status_code", None)
    if status is None:
        return None
    if status == 404:
        return "model_not_pulled", {}
    if status in (401, 403):
        return "auth_failed", {}
    if status == 429:
        retry_after_raw = None
        try:
            headers = getattr(response, "headers", None)
            if headers is not None:
                retry_after_raw = headers.get("Retry-After")
        except Exception:  # noqa: BLE001 — best-effort header read
            retry_after_raw = None
        extra: dict[str, object] = {}
        if retry_after_raw is not None:
            extra["retry_after"] = retry_after_raw
        return "rate_limited", extra
    return None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


_PROBE_TEXT = "probe"


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

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        ...

    def probe_dimension(self) -> int:
        """Return the dimensionality of vectors produced by this embedder.

        Implementations issue a single-text dry-run embed of a fixed
        probe string when no static lookup is available, then cache the
        result. This replaces the previous behavior where
        ``RemoteEmbedder.dimension`` silently returned a hardcoded
        ``1536`` fallback for unknown models and
        ``OllamaEmbedder.dimension`` returned ``0`` until the first
        embed call had succeeded — both bugs that produced
        dimension-mismatch failures on first ingest into a vector
        collection sized at the actual probed dimension.
        """
        ...


# ---------------------------------------------------------------------------
# SentenceTransformers provider
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Local embedder backed by the ``sentence-transformers`` library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
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

    @property
    def model_name(self) -> str:
        """Return the name of the loaded SentenceTransformer model."""
        return self._model_name

    def probe_dimension(self) -> int:
        """Return the embedding dimension; the loaded model reports it directly."""
        return self.dimension


# ---------------------------------------------------------------------------
# Remote HTTP provider (/v1/embeddings surface)
# ---------------------------------------------------------------------------

# Known dimensions for common remote embedding models.
_REMOTE_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class RemoteEmbedder:
    """Remote embedder that POSTs to an HTTP ``/v1/embeddings`` endpoint.

    Speaks the common embeddings JSON contract (``{"input": [...],
    "model": ...}`` → ``{"data": [{"index": N, "embedding": [...]}]}``)
    used by Ollama's ``/v1`` surface, vLLM, LocalAI, and widely-available
    hosted APIs. Every call is wall-clock bounded by ``timeout_seconds``
    with up to ``max_retries`` retries, so a stalled backend cannot block
    the MCP RPC indefinitely — on timeout or connection failure, callers
    receive an :class:`EmbedderUnavailableError` with a FIELD-14
    structured payload.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
        _client: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._api_key = api_key
        if _client is not None:
            self._client = _client
        else:
            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            self._client = httpx.Client(
                timeout=timeout_seconds,
                headers=headers,
            )
        self._dimension_cache: int | None = None

    def _call_embed(self, texts: list[str], *, phase: str) -> list[list[float]]:
        """POST to the embeddings endpoint, converting transport errors.

        ``phase`` identifies the call site (``embed_query`` /
        ``embed_batch``) and ends up in the structured-error payload.
        Retries are bounded by ``max_retries``; on final transport
        failure, raises :class:`EmbedderUnavailableError`. HTTP 4xx
        responses we know how to classify (404 model-not-pulled,
        401/403 auth, 429 rate-limited) are also surfaced as
        :class:`EmbedderUnavailableError` with an appropriate error
        token so MCP tools can return a structured payload instead of
        propagating an opaque ``httpx.HTTPStatusError``.
        """
        url = self._base_url + "/embeddings"
        body = {"input": texts, "model": self._model_name}
        started = time.monotonic()
        total_attempts = self._max_retries + 1
        for attempt in range(total_attempts):
            try:
                response = self._client.post(url, json=body)
                response.raise_for_status()
                parsed = response.json()
                break
            except httpx.HTTPStatusError as exc:
                classified = _classify_http_status_error(exc)
                if classified is None:
                    raise
                error_token, extra = classified
                latency_ms = int((time.monotonic() - started) * 1000)
                raise EmbedderUnavailableError(
                    error=error_token,
                    model=self._model_name,
                    phase=phase,
                    latency_ms=latency_ms,
                    detail=type(exc).__name__,
                    extra=extra,
                ) from exc
            except BaseException as exc:  # noqa: BLE001 — classify below
                error_token = _classify_httpx_transport_error(exc)
                if error_token is None:
                    raise
                if attempt + 1 < total_attempts:
                    continue
                latency_ms = int((time.monotonic() - started) * 1000)
                raise EmbedderUnavailableError(
                    error=error_token,
                    model=self._model_name,
                    phase=phase,
                    latency_ms=latency_ms,
                    detail=type(exc).__name__,
                ) from exc

        data = parsed.get("data", [])
        sorted_data = sorted(data, key=lambda d: d["index"])
        return [item["embedding"] for item in sorted_data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed *texts* via the remote endpoint."""
        return self._call_embed(texts, phase="embed_batch")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._call_embed([text], phase="embed_query")[0]

    def probe_dimension(self) -> int:
        """Return this embedder's vector dimension, probing if necessary.

        Resolution order:

        1. Cached value from a previous probe or successful embed.
        2. Static lookup in :data:`_REMOTE_EMBEDDING_DIMENSIONS` for
           well-known OpenAI-compatible models (avoids an HTTP call).
        3. Live dry-run embed of the literal probe string ``"probe"``;
           the resulting vector's length is cached and returned.
        """
        if self._dimension_cache is not None:
            return self._dimension_cache
        known = _REMOTE_EMBEDDING_DIMENSIONS.get(self._model_name)
        if known is not None:
            self._dimension_cache = known
            return known
        vectors = self._call_embed([_PROBE_TEXT], phase="probe_dimension")
        if not vectors or not vectors[0]:
            raise EmbedderUnavailableError(
                error="probe_failed",
                model=self._model_name,
                phase="probe_dimension",
                latency_ms=0,
                detail="empty embedding response",
            )
        dim = len(vectors[0])
        self._dimension_cache = dim
        return dim

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension for the configured model.

        Returns the static value from :data:`_REMOTE_EMBEDDING_DIMENSIONS`
        when the model is known, otherwise probes the live endpoint via
        :meth:`probe_dimension`. The previous hardcoded 1536 fallback —
        which silently produced dimension mismatches for any non-OpenAI
        model — has been removed.
        """
        if self._dimension_cache is not None:
            return self._dimension_cache
        known = _REMOTE_EMBEDDING_DIMENSIONS.get(self._model_name)
        if known is not None:
            self._dimension_cache = known
            return known
        return self.probe_dimension()

    @property
    def model_name(self) -> str:
        """Return the name of the remote embedding model."""
        return self._model_name


# ---------------------------------------------------------------------------
# Ollama native provider (/api/embed)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaEmbedder:
    """Embedder targeting Ollama's native ``/api/embed`` endpoint.

    Unlike :class:`RemoteEmbedder` (which speaks the OpenAI-compatible
    ``/v1/embeddings`` contract and requires a ``/v1`` suffix plus a
    Bearer API key), this provider:

    * POSTs to ``<base_url>/api/embed`` with ``{"model": ..., "input":
      [...]}``;
    * parses ``{"embeddings": [[...]]}`` responses;
    * needs **no** API key (Ollama does not authenticate locally).

    Every call is wall-clock bounded by ``timeout_seconds`` with up to
    ``max_retries`` retries; transport failures surface as
    :class:`EmbedderUnavailableError` with the same FIELD-14 payload
    shape used by :class:`RemoteEmbedder`.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
        _client: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        if _client is not None:
            self._client = _client
        else:
            self._client = httpx.Client(timeout=timeout_seconds)
        self._dimension_cache: int | None = None

    def _call_embed(self, texts: list[str], *, phase: str) -> list[list[float]]:
        """POST to ``/api/embed`` and decode the Ollama response shape.

        Transport errors (timeout / network) become
        :class:`EmbedderUnavailableError` after retries are exhausted.
        Ollama-side HTTP 4xx responses we recognise (404 model-not-pulled,
        401/403 auth, 429 rate-limited) are also classified as
        :class:`EmbedderUnavailableError` so the MCP layer sees a
        structured payload instead of an opaque ``httpx.HTTPStatusError``.
        """
        url = self._base_url + "/api/embed"
        body = {"model": self._model_name, "input": texts}
        started = time.monotonic()
        total_attempts = self._max_retries + 1
        for attempt in range(total_attempts):
            try:
                response = self._client.post(url, json=body)
                response.raise_for_status()
                parsed = response.json()
                break
            except httpx.HTTPStatusError as exc:
                classified = _classify_http_status_error(exc)
                if classified is None:
                    raise
                error_token, extra = classified
                latency_ms = int((time.monotonic() - started) * 1000)
                raise EmbedderUnavailableError(
                    error=error_token,
                    model=self._model_name,
                    phase=phase,
                    latency_ms=latency_ms,
                    detail=type(exc).__name__,
                    extra=extra,
                ) from exc
            except BaseException as exc:  # noqa: BLE001 — classify below
                error_token = _classify_httpx_transport_error(exc)
                if error_token is None:
                    raise
                if attempt + 1 < total_attempts:
                    continue
                latency_ms = int((time.monotonic() - started) * 1000)
                raise EmbedderUnavailableError(
                    error=error_token,
                    model=self._model_name,
                    phase=phase,
                    latency_ms=latency_ms,
                    detail=type(exc).__name__,
                ) from exc

        embeddings = parsed.get("embeddings", [])
        if embeddings and self._dimension_cache is None:
            self._dimension_cache = len(embeddings[0])
        return list(embeddings)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed *texts* via Ollama's ``/api/embed`` endpoint."""
        if not texts:
            return []
        return self._call_embed(texts, phase="embed_batch")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._call_embed([text], phase="embed_query")[0]

    def probe_dimension(self) -> int:
        """Return this embedder's vector dimension, probing if necessary.

        Issues a single-text dry-run embed of the literal probe string
        ``"probe"`` when the dimension cache is empty, populating it
        from the response. Replaces the previous behaviour where
        :attr:`dimension` returned ``0`` until the first regular embed
        had succeeded — guaranteeing a dimension mismatch when a
        ChromaDB collection was sized at 0.
        """
        if self._dimension_cache is not None:
            return self._dimension_cache
        vectors = self._call_embed([_PROBE_TEXT], phase="probe_dimension")
        if not vectors or not vectors[0]:
            raise EmbedderUnavailableError(
                error="probe_failed",
                model=self._model_name,
                phase="probe_dimension",
                latency_ms=0,
                detail="empty embedding response",
            )
        # ``_call_embed`` populates ``_dimension_cache`` on success but
        # guard explicitly for the test-mock case where the body uses a
        # different shape.
        if self._dimension_cache is None:
            self._dimension_cache = len(vectors[0])
        return self._dimension_cache

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, probing on first access if needed.

        Cached after the first successful embed or probe. Unlike the
        previous behaviour (which returned ``0`` until the first
        regular embed completed), accessing this property now triggers
        a probe call when no cached value is available, so callers
        sizing a ChromaDB collection get the real dimension up front.
        """
        if self._dimension_cache is None:
            self.probe_dimension()
        # ``probe_dimension`` populates the cache; assert non-None for
        # the type-checker.
        assert self._dimension_cache is not None
        return self._dimension_cache

    @property
    def model_name(self) -> str:
        """Return the Ollama model identifier."""
        return self._model_name


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_embedder(config: Settings) -> Embedder:
    """Instantiate the appropriate :class:`Embedder` based on *config*.

    Raises:
        ValueError: If the remote provider is selected but no API key or
            base URL is set.
    """
    return _build_embedder(
        provider=config.embedding_provider,
        model=config.embedding_model,
        base_url=config.embed_base_url,
        api_key=config.embed_api_key,
        timeout_seconds=config.embed_timeout_seconds,
        max_retries=config.embed_max_retries,
    )


def create_embedder_for_model(
    config: Settings, model_name: str | None
) -> Embedder:
    """Instantiate an :class:`Embedder` matching *model_name*.

    Used at query time to guarantee the query vector is produced by the
    same model that originally ingested the vectorstore.  When
    ``model_name`` is None or equals the configured default, the regular
    configured embedder is returned; otherwise, a fresh embedder using
    the same provider/base_url/api_key but overriding the model name
    is created.
    """
    if not model_name or model_name == config.embedding_model:
        return create_embedder(config)
    return _build_embedder(
        provider=config.embedding_provider,
        model=model_name,
        base_url=config.embed_base_url,
        api_key=config.embed_api_key,
        timeout_seconds=config.embed_timeout_seconds,
        max_retries=config.embed_max_retries,
    )


def _build_embedder(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> Embedder:
    """Internal builder shared by :func:`create_embedder` variants."""
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model)
    if provider == "ollama":
        return OllamaEmbedder(
            model_name=model,
            base_url=base_url or DEFAULT_OLLAMA_BASE_URL,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    if provider == "remote":
        if not api_key:
            raise ValueError("AGENT_KB_EMBED_API_KEY required for remote embeddings")
        if not base_url:
            raise ValueError(
                "AGENT_KB_EMBED_BASE_URL required for remote embeddings "
                "(e.g. http://localhost:11434/v1 for Ollama's OpenAI-compat endpoint)"
            )
        return RemoteEmbedder(
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    # Should be unreachable due to the Literal type constraint, but guard anyway.
    raise ValueError(f"Unknown embedding provider: {provider}")
