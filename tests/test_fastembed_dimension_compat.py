"""Phase 5 — FastembedEmbedder dimension compat smoke test.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[5].tasks[2,3] — add 'fastembed' provider;
        Phase 0 dimension-probe semantics still apply.

Runs only when ``fastembed`` is importable (the
``[embed-local-onnx]`` extra is installed); otherwise the module is
skipped entirely so the rest of the test suite stays green on a
default install.
"""

from __future__ import annotations

import pytest

# Skip the entire module when fastembed isn't installed. The Phase 5
# default install does NOT carry fastembed; users who want it
# install via ``pip install agent-knowledgebase[embed-local-onnx]``.
fastembed = pytest.importorskip("fastembed")  # noqa: F401

from agent_knowledgebase.services.embeddings import FastembedEmbedder  # noqa: E402


# Standard MiniLM-L6-v2 dimensionality. The model name fastembed
# uses for its bundled MiniLM family varies by version — try the
# common identifier first and fall back to BAAI/bge-small-en-v1.5
# (also 384-dim) for environments where the MiniLM bundle isn't
# pre-listed. Both are 384-dim.
_CANDIDATE_384_DIM_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
)


def _build_first_available_embedder() -> FastembedEmbedder:
    last_exc: Exception | None = None
    for model in _CANDIDATE_384_DIM_MODELS:
        try:
            return FastembedEmbedder(model_name=model)
        except Exception as exc:  # noqa: BLE001 — try the next candidate
            last_exc = exc
    pytest.skip(
        f"No 384-dim fastembed model available locally; last error: {last_exc!r}"
    )


def test_fastembed_minilm_class_reports_384_dim() -> None:
    """A MiniLM-class fastembed model probes to 384 dimensions."""
    emb = _build_first_available_embedder()
    dim = emb.probe_dimension()
    assert dim == 384, (
        f"Expected MiniLM-class fastembed model to be 384-dim, got {dim}"
    )


def test_fastembed_dimension_property_caches_after_probe() -> None:
    """``dimension`` cached after first probe — no re-probe on access."""
    emb = _build_first_available_embedder()
    first = emb.probe_dimension()
    second = emb.dimension  # property access; should hit cache
    assert first == second
    # And the cache itself is set:
    assert emb._dimension_cache == first  # noqa: SLF001 — by design


def test_fastembed_embedder_version_marks_int8() -> None:
    """``embedder_version`` includes the int8 quantization marker so
    the Phase 5 mixed-version rejection can distinguish fastembed-int8
    from sentence-transformers-fp32 even at the same model_name."""
    emb = _build_first_available_embedder()
    version = emb.embedder_version
    assert version.startswith("fastembed/")
    assert version.endswith("-int8"), (
        f"fastembed embedder_version must mark int8 quantization, got {version!r}"
    )


def test_fastembed_embed_returns_list_of_lists() -> None:
    """``embed`` returns a Python list-of-lists (not numpy arrays)."""
    emb = _build_first_available_embedder()
    vectors = emb.embed(["hello", "world"])
    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])
