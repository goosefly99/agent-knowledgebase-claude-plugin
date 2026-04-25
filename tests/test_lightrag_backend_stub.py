"""Phase 6 tests for the :class:`LightRAGBackend` stub.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[6] (Phase 6 — LightRAGBackend, deferred)

Phase 6 ships only the stub class and a design doc — see
``docs/lightrag_backend.md``. These tests pin the stub's contract so
the surface stays activation-ready:

* Construction succeeds (no NotImplementedError at import / __init__).
* :meth:`info` returns all 5 probe-4 keys with ``None`` values plus a
  ``backend='lightrag'`` discriminator + ``status='unavailable'``.
* :meth:`health_check` returns ``status='unavailable'``.
* Non-trivial methods raise :class:`NotImplementedError` with the
  canonical activation message that mentions
  ``AGENT_KB_BACKEND=lightrag``.
* The ``get_backend()`` factory branch for ``kb_backend='lightrag'``
  now CONSTRUCTS a :class:`LightRAGBackend` (Phase 6 promoted the
  branch from raise-NotImplementedError to a stub instance).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.backends import RetrieverBackend, get_backend
from agent_knowledgebase.backends.lightrag_backend import LightRAGBackend
from agent_knowledgebase.config import Settings


# ---------------------------------------------------------------------------
# Probe-4 frozen keys per spec.constraints[4] (validation finding f-20).
# ---------------------------------------------------------------------------


_PROBE4_BACKEND_INFO_KEYS = frozenset(
    {
        "source_type",
        "uri",
        "dedup_key",
        "page_id",
        "dominant_embedding_model",
    }
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def lightrag_settings(tmp_path: Path) -> Settings:
    """Build a Settings pointing at a tmp_path saves dir w/ kb_backend=lightrag."""
    saves = tmp_path / "saves"
    saves.mkdir()
    return Settings(saves_dir=saves, kb_backend="lightrag").resolve_paths()


# ---------------------------------------------------------------------------
# Importability + construction
# ---------------------------------------------------------------------------


def test_lightrag_backend_importable() -> None:
    """The module imports cleanly and exports the class."""
    from agent_knowledgebase.backends.lightrag_backend import LightRAGBackend as LRG

    assert LRG is LightRAGBackend


def test_lightrag_backend_constructs_without_raising(
    lightrag_settings: Settings,
) -> None:
    """``LightRAGBackend(settings)`` constructs without NotImplementedError.

    Phase 6 explicitly promoted the factory branch from
    raise-on-construct to construct-but-stub-methods so consumers can
    probe info() / health_check() without crashing.
    """
    backend = LightRAGBackend(lightrag_settings)
    # Stored constructor args round-trip.
    assert backend._settings is lightrag_settings
    assert backend._service is None


def test_lightrag_backend_satisfies_runtime_checkable_protocol(
    lightrag_settings: Settings,
) -> None:
    """The stub satisfies the runtime-checkable RetrieverBackend Protocol.

    Pinning this here means a future activation that accidentally
    drops a method (e.g. renames ``info`` to ``status``) breaks loudly.
    """
    backend = LightRAGBackend(lightrag_settings)
    assert isinstance(backend, RetrieverBackend), (
        "LightRAGBackend stub no longer satisfies the RetrieverBackend "
        "Protocol — check that all 7 methods (index, query, search, "
        "delete, info, count, health_check) are still present."
    )


# ---------------------------------------------------------------------------
# Probe-4 contract — info() must return all 5 keys with None values
# ---------------------------------------------------------------------------


def test_lightrag_backend_info_returns_full_probe4_union(
    lightrag_settings: Settings,
) -> None:
    """:meth:`info` returns all 5 probe-4 keys (None) + backend discriminator.

    Probe-4 contract (validation finding f-20): the 5 keys MUST be
    present, NOT omitted, even when the backend has no data. The
    stub returns None for every probe-4 key plus a backend tag and a
    ``status='unavailable'`` flag so consumers can distinguish a
    non-activated backend from an empty-but-active one.
    """
    backend = LightRAGBackend(lightrag_settings)
    info = backend.info(kb_id="any-kb")

    missing = _PROBE4_BACKEND_INFO_KEYS - set(info)
    assert not missing, (
        f"LightRAGBackend.info() omitted probe-4 keys: {sorted(missing)} — "
        f"validation finding f-20 requires all 5 keys present (None when "
        f"inapplicable, NEVER omitted). Got: {sorted(info.keys())}"
    )
    for key in _PROBE4_BACKEND_INFO_KEYS:
        assert info[key] is None, (
            f"expected info()[{key!r}] == None on stub, got {info[key]!r}"
        )

    assert info["backend"] == "lightrag"
    assert info["status"] == "unavailable"
    assert info.get("reason") == "stub_not_activated"


# ---------------------------------------------------------------------------
# health_check returns the deferred-status payload
# ---------------------------------------------------------------------------


def test_lightrag_backend_health_check_reports_unavailable(
    lightrag_settings: Settings,
) -> None:
    """``health_check()`` reports the deferred status without raising."""
    backend = LightRAGBackend(lightrag_settings)
    health = backend.health_check()

    assert health["backend"] == "lightrag"
    assert health["status"] == "unavailable"
    assert health["spec_version"] == "2.1"
    assert health["reason"] == "stub_not_activated"


# ---------------------------------------------------------------------------
# Non-trivial methods raise the canonical activation message
# ---------------------------------------------------------------------------


def test_lightrag_backend_index_raises_with_activation_message(
    lightrag_settings: Settings,
) -> None:
    """:meth:`index` raises NotImplementedError mentioning AGENT_KB_BACKEND=lightrag.

    The activation message lists the three preconditions (sidecar URL,
    env var, per-method implementation) so an operator who hits the
    raise sees exactly what's missing.
    """
    backend = LightRAGBackend(lightrag_settings)
    with pytest.raises(NotImplementedError, match="AGENT_KB_BACKEND=lightrag"):
        backend.index(kb_id="x", documents=[])


def test_lightrag_backend_query_raises(lightrag_settings: Settings) -> None:
    """:meth:`query` raises with the canonical activation message."""
    backend = LightRAGBackend(lightrag_settings)
    with pytest.raises(NotImplementedError, match="docs/lightrag_backend.md"):
        backend.query(kb_id="x", text="t", top_k=1)


def test_lightrag_backend_search_raises(lightrag_settings: Settings) -> None:
    """:meth:`search` raises with the canonical activation message."""
    backend = LightRAGBackend(lightrag_settings)
    with pytest.raises(NotImplementedError, match="docs/lightrag_backend.md"):
        backend.search(kb_id="x", text="t", top_k=1)


def test_lightrag_backend_delete_raises(lightrag_settings: Settings) -> None:
    """:meth:`delete` raises with the canonical activation message."""
    backend = LightRAGBackend(lightrag_settings)
    with pytest.raises(NotImplementedError, match="AGENT_KB_LIGHTRAG_URL"):
        backend.delete(kb_id="x", ids=["a"])


def test_lightrag_backend_count_raises(lightrag_settings: Settings) -> None:
    """:meth:`count` raises with the canonical activation message."""
    backend = LightRAGBackend(lightrag_settings)
    with pytest.raises(NotImplementedError, match="LightRAGBackend"):
        backend.count(kb_id="x")


# ---------------------------------------------------------------------------
# Factory promotion — get_backend() now CONSTRUCTS a LightRAGBackend
# ---------------------------------------------------------------------------


def test_get_backend_factory_constructs_lightrag_in_phase6(tmp_path: Path) -> None:
    """``get_backend(Settings(kb_backend='lightrag'))`` returns a
    :class:`LightRAGBackend` instance.

    Phase 6 promoted this branch from
    ``raise NotImplementedError("...Phase 6 deferred...")`` to
    construction-of-stub-class. The construction must NOT raise — the
    raises now live on the per-method bodies (index/query/search/
    delete/count) and surface lazily on actual use.
    """
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(saves_dir=saves, kb_backend="lightrag").resolve_paths()
    backend = get_backend(settings)
    assert isinstance(backend, LightRAGBackend)
    assert isinstance(backend, RetrieverBackend)
