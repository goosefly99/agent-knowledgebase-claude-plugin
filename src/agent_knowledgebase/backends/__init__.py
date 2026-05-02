"""agent_knowledgebase.backends — single chromadb retrieval backend.

The v2.1 redesign's RetrieverBackend abstraction is retained as a
structural seam, but the v0.13.0 cut collapses to a single
implementation (ChromadbBackend). Removing the per-KB routing
machinery (.migrated_to sentinel, kb_backend_per_kb) keeps the
abstraction without the unused dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


@runtime_checkable
class RetrieverBackend(Protocol):
    def index(self, *, kb_id: str, documents: list[dict[str, Any]]) -> None: ...
    def query(self, *, kb_id: str, text: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    def search(self, *, kb_id: str, text: str, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    def delete(self, *, kb_id: str, ids: list[str] | None = None, source_id: str | None = None) -> None: ...
    def info(self, *, kb_id: str) -> dict[str, Any]: ...
    def count(self, *, kb_id: str) -> int: ...
    def health_check(self) -> dict[str, Any]: ...


def get_backend(
    settings: "Settings",
    *,
    service: "KnowledgebaseService | None" = None,
    kb_id: str | None = None,  # accepted for callsite-compat; ignored
) -> RetrieverBackend:
    """Return the single ChromadbBackend instance for the given service."""
    from .chromadb_backend import ChromadbBackend

    return ChromadbBackend(settings, service=service)


__all__ = ["RetrieverBackend", "get_backend"]
