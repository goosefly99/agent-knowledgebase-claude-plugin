"""LightRAG-backed :class:`RetrieverBackend` (Phase 6 stub, DEFERRED).

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[6] — LightRAGBackend (deferred, conditional)
        architecture.components[1] — AGENT_KB_BACKEND feature flag

Phase 6 status: STUB ONLY. The class implements the full
:class:`agent_knowledgebase.backends.RetrieverBackend` Protocol surface
so the factory branch in ``backends/__init__.py`` can construct a
LightRAGBackend without raising at import time, but every non-trivial
method raises :class:`NotImplementedError` with a clear activation
message. Activation requires:

1. A running LightRAG sidecar reachable at ``AGENT_KB_LIGHTRAG_URL``
   (defaults to ``http://localhost:9621``). The reference deployment
   is the ``HKUDS/LightRAG`` docker image — see
   ``docs/lightrag_backend.md`` for a ``docker-compose.yml`` snippet.
2. Operator opt-in via ``AGENT_KB_BACKEND=lightrag`` (or per-KB via
   ``AGENT_KB_BACKEND_PER_KB='kb1=lightrag'``).
3. Promoting this stub by replacing every ``NotImplementedError`` with
   the actual REST forwarding logic per ``docs/lightrag_backend.md``.

Why the stub is preserved instead of deleted
---------------------------------------------

The factory in ``backends/__init__.py`` previously raised
``NotImplementedError`` directly on the ``'lightrag'`` branch — that
shape forced every caller (including read-only health probes) to
handle the construction-time failure. Phase 6 promotes the branch to
"construct a LightRAGBackend stub that surfaces failures lazily on
the methods that would actually need a live sidecar." This means:

* :meth:`info` returns the full probe-4 contract with ``None`` values
  + a backend discriminator + ``status='unavailable'`` so consumers
  treat the KB as "present but not queryable" instead of crashing on
  a missing key.
* :meth:`health_check` returns ``{"backend": "lightrag", "status":
  "unavailable", ...}`` so a future ``kb_health`` MCP tool can report
  the deferred status without raising.
* All other methods (:meth:`index`, :meth:`query`, :meth:`search`,
  :meth:`delete`, :meth:`count`) raise :class:`NotImplementedError`
  with the canonical activation message defined in
  :data:`_ACTIVATION_MESSAGE`.

This preserves the probe-4 contract (validation finding f-20) for a
non-activated backend AND reserves the contract surface so future
activation has a clear landing place — the factory branch, the test
file, and the docs already exist; only the per-method bodies need to
be written.

Adoption-signal gate
--------------------

LightRAG support is conditional on adoption signal. If no operator
opts in within 6 months of v0.11.0 GA, this stub may be deleted to
reduce maintenance surface. See ``docs/lightrag_backend.md`` for the
operator-facing trigger conditions and the activation checklist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


_BACKEND_NAME = "lightrag"
_SPEC_VERSION = "2.1"
_SPEC_ID = "70ab2170-381a-4657-bcd1-28a40c6f369b"

# Canonical activation message raised from every non-trivial method.
# Surfaces the three preconditions the operator must satisfy before
# this stub can be promoted to a live backend. Kept as a module-level
# constant so the test file can assert the message verbatim and the
# docs page can quote it without drift.
_ACTIVATION_MESSAGE = (
    "LightRAGBackend is a deferred Phase 6 deliverable. Activation "
    "requires (a) a running LightRAG sidecar at AGENT_KB_LIGHTRAG_URL "
    "(default http://localhost:9621), (b) AGENT_KB_BACKEND=lightrag, "
    "and (c) implementation of this backend per docs/lightrag_backend.md. "
    "The stub exists to reserve the contract surface."
)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class LightRAGBackend:
    """Deferred LightRAG :class:`RetrieverBackend` (Phase 6 stub).

    Construction parameters
    -----------------------
    settings:
        The resolved :class:`Settings` instance the owning service was
        built from. Stored for forward-compat — once the stub is
        activated the constructor will read ``AGENT_KB_LIGHTRAG_URL``
        and any LightRAG-specific Settings fields from here.
    service:
        Optional back-reference to the owning
        :class:`KnowledgebaseService`. Mirrors the
        :class:`ChromadbBackend` / :class:`MarkdownWikiBackend`
        construction signature so the factory in
        ``backends/__init__.py`` can dispatch uniformly across all
        three backends. Currently unused — the stub only ever returns
        from :meth:`info` and :meth:`health_check` without touching
        per-KB plumbing.

    All non-trivial methods raise :class:`NotImplementedError` with
    :data:`_ACTIVATION_MESSAGE`. :meth:`info` and :meth:`health_check`
    are intentionally non-raising so consumers can probe a
    non-activated backend without crashing.
    """

    def __init__(
        self,
        settings: "Settings",
        *,
        service: "KnowledgebaseService | None" = None,
    ) -> None:
        self._settings = settings
        self._service = service

    # ------------------------------------------------------------------
    # RetrieverBackend Protocol — write paths (raise on stub)
    # ------------------------------------------------------------------

    def index(
        self,
        *,
        kb_id: str,
        documents: list[dict[str, Any]],
    ) -> None:
        """Add documents — NOT IMPLEMENTED in the stub.

        Activation will forward each document to LightRAG's
        ``/insert`` REST endpoint (or the streaming bulk-insert path).
        See ``docs/lightrag_backend.md`` "REST API contract" for the
        full endpoint surface.
        """
        raise NotImplementedError(_ACTIVATION_MESSAGE)

    def query(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector / hybrid query — NOT IMPLEMENTED in the stub.

        Activation will POST to LightRAG's ``/query`` endpoint with
        ``mode='hybrid'`` (LightRAG's graph-aware retrieval default)
        and reshape the response into the Protocol's
        :class:`agent_knowledgebase.services.query.SearchResult`-shaped
        list of dicts.
        """
        raise NotImplementedError(_ACTIVATION_MESSAGE)

    def search(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword search — NOT IMPLEMENTED in the stub.

        Activation will POST to LightRAG's ``/query`` endpoint with
        ``mode='naive'`` (LightRAG's keyword-only retrieval mode) so
        the keyword path stays distinct from the graph-aware
        :meth:`query` path.
        """
        raise NotImplementedError(_ACTIVATION_MESSAGE)

    def delete(
        self,
        *,
        kb_id: str,
        ids: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        """Delete documents — NOT IMPLEMENTED in the stub.

        Activation will call LightRAG's ``/delete`` endpoint by id.
        """
        raise NotImplementedError(_ACTIVATION_MESSAGE)

    def count(self, *, kb_id: str) -> int:
        """Stored chunk count — NOT IMPLEMENTED in the stub.

        Activation will read LightRAG's ``/info`` endpoint and surface
        the entity / relation / chunk count triple.
        """
        raise NotImplementedError(_ACTIVATION_MESSAGE)

    # ------------------------------------------------------------------
    # RetrieverBackend Protocol — metadata / health (non-raising)
    # ------------------------------------------------------------------

    def info(self, *, kb_id: str) -> dict[str, Any]:
        """Backend health + per-KB metadata.

        Returns the v0.6.0 probe-4 contract fields with ``None``
        values plus a ``backend`` discriminator and a
        ``status='unavailable'`` flag so the data-etl-orchestrator
        probe-4 introspection step does NOT crash on a non-activated
        backend.

        Probe-4 contract (validation finding f-20): ``source_type``,
        ``uri``, ``dedup_key``, ``page_id``, ``dominant_embedding_model``
        MUST all be present (with value ``None``), NOT omitted.
        """
        return {
            # Probe-4 frozen keys (None — stub has no data).
            "source_type": None,
            "uri": None,
            "dedup_key": None,
            "page_id": None,
            "dominant_embedding_model": None,
            # Backend diagnostics (additive — surfaces deferred status).
            "backend": _BACKEND_NAME,
            "spec_version": _SPEC_VERSION,
            "status": "unavailable",
            "reason": "stub_not_activated",
        }

    def health_check(self) -> dict[str, Any]:
        """Backend-wide health probe (no kb_id).

        Returns ``{"backend": "lightrag", "status": "unavailable",
        "spec_version": "2.1", "reason": "stub_not_activated"}`` so a
        future ``kb_health`` MCP tool can introspect the deferred
        status without raising. Once activation lands the body will
        round-trip ``GET /health`` against the LightRAG sidecar and
        flip ``status`` to ``"ok"`` / ``"degraded"`` based on the
        response.
        """
        return {
            "backend": _BACKEND_NAME,
            "status": "unavailable",
            "spec_version": _SPEC_VERSION,
            "reason": "stub_not_activated",
        }


__all__ = ["LightRAGBackend"]
