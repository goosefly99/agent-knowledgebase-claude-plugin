"""Phase 2 Protocol-conformance tests for every concrete RetrieverBackend.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[2] — RetrieverBackend abstraction

For every concrete backend (currently just :class:`ChromadbBackend`), the
test suite asserts:

  1. The backend instance satisfies the runtime-checkable
     :class:`agent_knowledgebase.backends.RetrieverBackend` Protocol.
  2. Each Protocol method exists with the expected signature
     (parameter names + kwarg-only flags) — caught via
     :func:`inspect.signature` so a future refactor that quietly
     renames a kwarg surfaces as a Phase-2-contract regression rather
     than a downstream call-site failure.

Phase 3 will add a MarkdownWikiBackend variant and parameterize this
test across both backends — the parameterization machinery is in
``_BACKEND_FACTORIES`` to make that addition mechanical.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

import pytest

from agent_knowledgebase.backends import RetrieverBackend, get_backend
from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend
from agent_knowledgebase.backends.lightrag_backend import LightRAGBackend
from agent_knowledgebase.backends.markdown_backend import MarkdownWikiBackend
from agent_knowledgebase.config import Settings


# ---------------------------------------------------------------------------
# Backend factories — extended in Phase 3 with the markdown variant and
# in Phase 6 with the lightrag stub.
# ---------------------------------------------------------------------------


def _build_chromadb_backend(saves_dir: Path) -> RetrieverBackend:
    """Construct a ChromadbBackend pointing at ``saves_dir``.

    No ``service`` back-reference is supplied — tests that touch
    per-KB methods would need one, but this Protocol-conformance suite
    only inspects method signatures so a bare backend is sufficient.
    """
    settings = Settings(saves_dir=saves_dir)
    settings = settings.resolve_paths()
    return ChromadbBackend(settings)


def _build_markdown_backend(saves_dir: Path) -> RetrieverBackend:
    """Construct a MarkdownWikiBackend pointing at ``saves_dir``.

    Same Protocol-conformance contract as the chromadb variant — no
    service back-reference is needed for the surface-level checks below.
    """
    settings = Settings(saves_dir=saves_dir, kb_backend="markdown")
    settings = settings.resolve_paths()
    return MarkdownWikiBackend(settings)


def _build_lightrag_backend(saves_dir: Path) -> RetrieverBackend:
    """Construct a LightRAGBackend stub pointing at ``saves_dir``.

    Phase 6 promoted the lightrag factory branch from
    raise-NotImplementedError to a stub that constructs successfully;
    info() and health_check() return non-raising responses while the
    other Protocol methods raise lazily with the canonical activation
    message. The Protocol-conformance suite only inspects method
    signatures so the stub is fully exercised here.
    """
    settings = Settings(saves_dir=saves_dir, kb_backend="lightrag")
    settings = settings.resolve_paths()
    return LightRAGBackend(settings)


_BACKEND_FACTORIES: dict[str, Callable[[Path], RetrieverBackend]] = {
    "chromadb": _build_chromadb_backend,
    "markdown": _build_markdown_backend,
    "lightrag": _build_lightrag_backend,
}


# ---------------------------------------------------------------------------
# Expected Protocol method signatures.
# ---------------------------------------------------------------------------


# Each entry: method_name -> tuple of (param_name, kind) tuples in the
# order they appear on the Protocol. ``kind`` matches
# ``inspect.Parameter.kind``. We assert the public surface — the
# self-binding parameter is excluded since it differs between
# unbound-on-Protocol and bound-on-instance signatures.
_EXPECTED_SIGNATURES = {
    "index": (
        ("kb_id", inspect.Parameter.KEYWORD_ONLY),
        ("documents", inspect.Parameter.KEYWORD_ONLY),
    ),
    "query": (
        ("kb_id", inspect.Parameter.KEYWORD_ONLY),
        ("text", inspect.Parameter.KEYWORD_ONLY),
        ("top_k", inspect.Parameter.KEYWORD_ONLY),
        ("filters", inspect.Parameter.KEYWORD_ONLY),
    ),
    "search": (
        ("kb_id", inspect.Parameter.KEYWORD_ONLY),
        ("text", inspect.Parameter.KEYWORD_ONLY),
        ("top_k", inspect.Parameter.KEYWORD_ONLY),
        ("filters", inspect.Parameter.KEYWORD_ONLY),
    ),
    "delete": (
        ("kb_id", inspect.Parameter.KEYWORD_ONLY),
        ("ids", inspect.Parameter.KEYWORD_ONLY),
        ("source_id", inspect.Parameter.KEYWORD_ONLY),
    ),
    "info": (("kb_id", inspect.Parameter.KEYWORD_ONLY),),
    "count": (("kb_id", inspect.Parameter.KEYWORD_ONLY),),
    "health_check": (),
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", sorted(_BACKEND_FACTORIES))
def test_backend_satisfies_runtime_checkable_protocol(backend_name: str, tmp_path: Path) -> None:
    """Every concrete backend must satisfy ``isinstance(backend,
    RetrieverBackend)`` (the Protocol is decorated ``@runtime_checkable``).
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _BACKEND_FACTORIES[backend_name](saves_dir)
    assert isinstance(backend, RetrieverBackend), (
        f"Backend {backend_name!r} ({type(backend).__name__}) does not "
        f"satisfy RetrieverBackend Protocol — check that all required "
        f"methods are present and callable."
    )


@pytest.mark.parametrize("backend_name", sorted(_BACKEND_FACTORIES))
@pytest.mark.parametrize("method_name", sorted(_EXPECTED_SIGNATURES))
def test_backend_method_signatures_match_protocol(
    backend_name: str, method_name: str, tmp_path: Path
) -> None:
    """Each Protocol method exists on the concrete backend with the
    expected (param_name, kind) tuple.

    Catches accidental renames (e.g. ``kb_id`` -> ``knowledgebase_id``)
    that would otherwise only break at call sites.
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _BACKEND_FACTORIES[backend_name](saves_dir)

    method = getattr(backend, method_name, None)
    assert method is not None, (
        f"{type(backend).__name__} is missing required method {method_name!r}"
    )
    assert callable(method), f"{type(backend).__name__}.{method_name} exists but is not callable"

    sig = inspect.signature(method)
    expected = _EXPECTED_SIGNATURES[method_name]

    # On a bound method, ``self`` is already removed by inspect.signature.
    actual = tuple((name, param.kind) for name, param in sig.parameters.items())

    assert actual == expected, (
        f"{type(backend).__name__}.{method_name} signature mismatch.\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )


def test_get_backend_factory_returns_chromadb_for_default_settings(
    tmp_path: Path,
) -> None:
    """Default ``settings.kb_backend`` is ``'chromadb'`` -> ChromadbBackend."""
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    settings = Settings(saves_dir=saves_dir).resolve_paths()
    backend = get_backend(settings)
    assert isinstance(backend, ChromadbBackend)
    assert isinstance(backend, RetrieverBackend)


def test_get_backend_factory_returns_markdown_when_opted_in(
    tmp_path: Path,
) -> None:
    """``AGENT_KB_BACKEND=markdown`` instantiates :class:`MarkdownWikiBackend`.

    Phase 3 promoted the markdown branch of ``get_backend()`` from
    ``NotImplementedError`` to a real backend instance. The default
    remains ``'chromadb'`` (covered by
    :func:`test_get_backend_factory_returns_chromadb_for_default_settings`).
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    settings = Settings(saves_dir=saves_dir, kb_backend="markdown").resolve_paths()
    backend = get_backend(settings)
    assert isinstance(backend, MarkdownWikiBackend)
    assert isinstance(backend, RetrieverBackend)


def test_get_backend_factory_returns_lightrag_stub_in_phase6(
    tmp_path: Path,
) -> None:
    """Phase 6 promoted the ``'lightrag'`` factory branch from
    raise-NotImplementedError to constructing a :class:`LightRAGBackend`
    stub.

    Construction must NOT raise — the raises now live on the
    per-method bodies (index/query/search/delete/count) and surface
    lazily on actual use. info() and health_check() return non-raising
    responses (status='unavailable') so probe-4 introspection of a
    non-activated backend doesn't crash.
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    settings = Settings(saves_dir=saves_dir, kb_backend="lightrag").resolve_paths()
    backend = get_backend(settings)
    assert isinstance(backend, LightRAGBackend)
    assert isinstance(backend, RetrieverBackend)
    # health_check is one of the two non-raising stub methods.
    health = backend.health_check()
    assert health["backend"] == "lightrag"
    assert health["status"] == "unavailable"


def test_lightrag_backend_index_raises_with_activation_message(
    tmp_path: Path,
) -> None:
    """The lightrag stub's :meth:`index` raises NotImplementedError
    referencing ``AGENT_KB_BACKEND=lightrag`` so an operator who hits
    the raise sees exactly which env var to flip.
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _build_lightrag_backend(saves_dir)
    with pytest.raises(NotImplementedError, match="AGENT_KB_BACKEND=lightrag"):
        backend.index(kb_id="x", documents=[])


def test_lightrag_backend_query_raises_with_activation_message(
    tmp_path: Path,
) -> None:
    """Symmetric guard — :meth:`query` raises with the canonical message."""
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _build_lightrag_backend(saves_dir)
    with pytest.raises(NotImplementedError, match="docs/lightrag_backend.md"):
        backend.query(kb_id="x", text="t", top_k=1)


def test_chromadb_backend_health_check_reports_chromadb(tmp_path: Path) -> None:
    """``health_check()`` returns the canonical backend tag so a future
    ``kb_health`` MCP tool can introspect.
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _build_chromadb_backend(saves_dir)
    health = backend.health_check()
    assert health["backend"] == "chromadb"
    assert health["status"] == "ok"
    assert health["spec_version"] == "2.1"
    assert health["vectorstore_provider"] == "chromadb"


def test_chromadb_backend_per_kb_methods_raise_clearly_without_service(
    tmp_path: Path,
) -> None:
    """A backend constructed without a service back-reference must
    raise a clear ``RuntimeError`` (not an opaque AttributeError) when
    a per-kb method is invoked.
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _build_chromadb_backend(saves_dir)
    with pytest.raises(RuntimeError, match="without a KnowledgebaseService"):
        backend.info(kb_id="missing-kb")


def test_chromadb_backend_query_accepts_valid_filters_kwarg(tmp_path: Path) -> None:
    """Phase A fix: ChromadbBackend.query no longer raises NotImplementedError
    when a valid ``filters`` dict is supplied.  The structural validator runs
    first; a valid dict proceeds to the service.  Constructing the backend
    without a service still raises RuntimeError (existing contract) — the
    test confirms the old NotImplementedError is gone and the new path raises
    RuntimeError (no service) instead of NotImplementedError (filter rejected).
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _build_chromadb_backend(saves_dir)
    # Valid filter dict: structural validation passes; RuntimeError (no
    # service) is raised before any chromadb call — NOT NotImplementedError.
    with pytest.raises(RuntimeError, match="without a KnowledgebaseService"):
        backend.query(kb_id="x", text="t", top_k=1, filters={"a": 1})


def test_chromadb_backend_query_rejects_invalid_filter_dict(tmp_path: Path) -> None:
    """Phase A: structural filter validation rejects non-string keys and
    non-primitive values before any chromadb or service call.
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _build_chromadb_backend(saves_dir)
    with pytest.raises((ValueError, TypeError)):
        backend.query(kb_id="x", text="t", top_k=1, filters={123: "v"})  # type: ignore[dict-item]


def test_chromadb_backend_search_rejects_non_none_filters(tmp_path: Path) -> None:
    """I-3 fix: ChromadbBackend.search raises ValueError on non-None filters.

    The FTS/wiki path has no metadata-filter support; silently dropping the
    caller's filters was a misleading contract. ValueError is raised before
    any service or chromadb call so no KnowledgebaseService is needed.
    """
    saves_dir = tmp_path / "saves"
    saves_dir.mkdir()
    backend = _build_chromadb_backend(saves_dir)
    with pytest.raises(ValueError, match="does not honor filters"):
        backend.search(kb_id="x", text="t", top_k=1, filters={"a": 1})
