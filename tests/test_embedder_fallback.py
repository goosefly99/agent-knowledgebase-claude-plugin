"""Tests for the ollama → sentence-transformers auto-fallback in create_embedder.

Contract under test (services/embeddings.py):

* ``create_embedder(config)`` probes ollama when provider=="ollama".
  On EmbedderUnavailableError → swap to SentenceTransformerEmbedder and
  emit a single-line JSON to stderr containing OLLAMA_FALLBACK_STDERR_TOKEN.
* Successful probe → return the OllamaEmbedder unchanged.
* provider != "ollama" → no probe, no fallback.
* ``create_embedder_for_model(...)`` with explicit provider/base_url overrides
  calls _build_embedder directly — NO fallback even when provider=="ollama"
  and the probe would fail.
* If ST init fails after an ollama probe failure → re-raise the original
  EmbedderUnavailableError (caller is no worse off than the no-fallback path).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.services.embeddings import (
    OLLAMA_FALLBACK_STDERR_TOKEN,
    EmbedderUnavailableError,
    OllamaEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    create_embedder_for_model,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ollama_config(tmp_path: Path) -> Settings:
    """Return a Settings wired to ollama with the standard test model."""
    return Settings(
        saves_dir=tmp_path,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:8b",
        embed_base_url="http://ollama:11434",
    )


def _probe_unavailable_error() -> EmbedderUnavailableError:
    return EmbedderUnavailableError(
        error="embed_unreachable",
        model="qwen3-embedding:8b",
        phase="probe_dimension",
        latency_ms=42,
    )


def _stub_st_init(self: SentenceTransformerEmbedder, model_name: str = "all-MiniLM-L6-v2") -> None:
    """Minimal SentenceTransformerEmbedder.__init__ stub that avoids model loading.

    Sets only ``_model_name`` so isinstance checks pass and the object is
    usable as an identity token in tests, without hitting HuggingFace Hub.
    """
    self._model_name = model_name
    self._model = MagicMock()
    self._model.get_embedding_dimension.return_value = 384


# ---------------------------------------------------------------------------
# Test 1: create_embedder falls back to ST when ollama probe is unreachable
# ---------------------------------------------------------------------------


def test_create_embedder_falls_back_when_ollama_probe_unreachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Probe raises EmbedderUnavailableError → returned embedder is SentenceTransformerEmbedder.

    The OLLAMA_UNREACHABLE_FALLBACK_TO_SENTENCE_TRANSFORMERS token must
    appear in stderr inside a single-line JSON object.
    SentenceTransformerEmbedder.__init__ is stubbed to avoid HuggingFace
    network calls during this unit test.
    """
    # Prevent real httpx.Client construction inside OllamaEmbedder.__init__.
    monkeypatch.setattr(httpx, "Client", MagicMock())

    # Probe always fails with a structured unreachable error.
    monkeypatch.setattr(
        OllamaEmbedder,
        "probe_dimension",
        lambda self: (_ for _ in ()).throw(_probe_unavailable_error()),
    )

    # Stub ST init so we don't attempt to download/load the real model.
    monkeypatch.setattr(SentenceTransformerEmbedder, "__init__", _stub_st_init)

    config = _ollama_config(tmp_path)
    embedder = create_embedder(config)

    # Must have swapped to the local fallback embedder.
    assert isinstance(embedder, SentenceTransformerEmbedder), (
        f"Expected SentenceTransformerEmbedder, got {type(embedder).__name__}"
    )

    # Stderr must contain the fallback token inside valid JSON.
    captured = capsys.readouterr()
    assert OLLAMA_FALLBACK_STDERR_TOKEN in captured.err, (
        f"Expected '{OLLAMA_FALLBACK_STDERR_TOKEN}' in stderr, got: {captured.err!r}"
    )
    # Must be a complete, parseable JSON line.
    stderr_line = captured.err.strip()
    payload = json.loads(stderr_line)
    assert payload.get("error_code") == OLLAMA_FALLBACK_STDERR_TOKEN


# ---------------------------------------------------------------------------
# Test 2: create_embedder returns the OllamaEmbedder when probe succeeds
# ---------------------------------------------------------------------------


def test_create_embedder_returns_ollama_when_probe_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Successful probe → OllamaEmbedder is returned; no fallback stderr token."""
    monkeypatch.setattr(httpx, "Client", MagicMock())
    monkeypatch.setattr(OllamaEmbedder, "probe_dimension", lambda self: 4096)

    config = _ollama_config(tmp_path)
    embedder = create_embedder(config)

    assert isinstance(embedder, OllamaEmbedder), (
        f"Expected OllamaEmbedder, got {type(embedder).__name__}"
    )
    # No fallback token must appear in stderr.
    captured = capsys.readouterr()
    assert OLLAMA_FALLBACK_STDERR_TOKEN not in captured.err, (
        f"Unexpected fallback token in stderr: {captured.err!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: create_embedder does NOT probe ollama for sentence-transformers provider
# ---------------------------------------------------------------------------


def test_create_embedder_does_not_fall_back_for_sentence_transformers_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """provider='sentence-transformers' → no probe of OllamaEmbedder occurs.

    OllamaEmbedder.probe_dimension is monkeypatched to raise loudly so the
    test will error if it is ever called. SentenceTransformerEmbedder.__init__
    is stubbed to avoid HuggingFace network calls.
    """

    def _should_never_be_called(self: OllamaEmbedder) -> int:
        raise AssertionError(
            "OllamaEmbedder.probe_dimension must NOT be called for "
            "provider='sentence-transformers'"
        )

    monkeypatch.setattr(OllamaEmbedder, "probe_dimension", _should_never_be_called)
    # Stub ST init to avoid HuggingFace model loading in a unit test.
    monkeypatch.setattr(SentenceTransformerEmbedder, "__init__", _stub_st_init)

    config = Settings(
        saves_dir=tmp_path,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
    )
    embedder = create_embedder(config)

    assert isinstance(embedder, SentenceTransformerEmbedder), (
        f"Expected SentenceTransformerEmbedder, got {type(embedder).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 4: create_embedder does NOT probe ollama for openai provider
# ---------------------------------------------------------------------------


def test_create_embedder_does_not_fall_back_for_openai_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """provider='openai' → OllamaEmbedder.probe_dimension is never called.

    The returned embedder must be an OpenAIEmbedder.
    """

    def _should_never_be_called(self: OllamaEmbedder) -> int:
        raise AssertionError(
            "OllamaEmbedder.probe_dimension must NOT be called for provider='openai'"
        )

    monkeypatch.setattr(OllamaEmbedder, "probe_dimension", _should_never_be_called)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-test")
    # Prevent any real HTTP client construction.
    monkeypatch.setattr(httpx, "Client", MagicMock())

    config = Settings(
        saves_dir=tmp_path,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embed_base_url="https://example.invalid/v1",
    )
    embedder = create_embedder(config)

    assert isinstance(embedder, OpenAIEmbedder), (
        f"Expected OpenAIEmbedder, got {type(embedder).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 5: create_embedder_for_model does NOT fall back on an ollama snapshot
# ---------------------------------------------------------------------------


def test_create_embedder_for_model_does_not_fall_back_on_ollama_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-snapshot rebuild with explicit provider/base_url override never applies fallback.

    Even when probe_dimension would raise EmbedderUnavailableError, the
    returned object must still be an OllamaEmbedder — unreachability surfaces
    only when the caller invokes embed().
    """
    monkeypatch.setattr(httpx, "Client", MagicMock())
    monkeypatch.setattr(
        OllamaEmbedder,
        "probe_dimension",
        lambda self: (_ for _ in ()).throw(_probe_unavailable_error()),
    )

    # Use any non-matching config so create_embedder_for_model is forced to
    # build a fresh embedder rather than delegating to create_embedder.
    config = Settings(
        saves_dir=tmp_path,
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
    )

    embedder = create_embedder_for_model(
        config,
        "qwen3-embedding:8b",
        provider="ollama",
        base_url="http://ollama:11434",
    )

    assert isinstance(embedder, OllamaEmbedder), (
        f"Expected OllamaEmbedder (no fallback in per-snapshot rebuild), "
        f"got {type(embedder).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 6: create_embedder re-raises the original EmbedderUnavailableError
#         when both ollama probe and ST init fail
# ---------------------------------------------------------------------------


def test_fallback_preserves_original_error_when_st_init_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ST init also fails, the ORIGINAL EmbedderUnavailableError is re-raised.

    The caller must be no worse off than the no-fallback path: the original
    structured error payload (error, model, phase, latency_ms) is preserved.
    """
    monkeypatch.setattr(httpx, "Client", MagicMock())

    original_error = _probe_unavailable_error()

    monkeypatch.setattr(
        OllamaEmbedder,
        "probe_dimension",
        lambda self: (_ for _ in ()).throw(original_error),
    )

    def _st_init_fails(self: SentenceTransformerEmbedder, model_name: str = "all-MiniLM-L6-v2") -> None:
        raise RuntimeError("missing extra: pip install sentence-transformers")

    monkeypatch.setattr(SentenceTransformerEmbedder, "__init__", _st_init_fails)

    config = _ollama_config(tmp_path)

    with pytest.raises(EmbedderUnavailableError) as excinfo:
        create_embedder(config)

    # Must be the ORIGINAL error, not a wrapped RuntimeError.
    raised = excinfo.value
    assert raised is original_error, (
        "create_embedder must re-raise the original EmbedderUnavailableError "
        "when SentenceTransformerEmbedder init also fails"
    )
    payload = raised.to_payload()
    assert payload["error"] == "embed_unreachable"
    assert payload["model"] == "qwen3-embedding:8b"
    assert payload["phase"] == "probe_dimension"
    assert payload["latency_ms"] == 42
