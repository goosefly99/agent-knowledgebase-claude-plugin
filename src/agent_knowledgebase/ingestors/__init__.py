"""Ingestor protocol, shared helpers, and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import tiktoken

from agent_knowledgebase.models import Chunk, SourceType


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RawContent:
    """Raw extracted content from a source before chunking."""

    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkConfig:
    """Configuration for token-based text chunking."""

    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64  # tokens


# ---------------------------------------------------------------------------
# Token-based chunking helper
# ---------------------------------------------------------------------------

_ENCODING = tiktoken.get_encoding("cl100k_base")


def token_chunk(contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
    """Split *contents* into overlapping token-window :class:`Chunk` objects.

    Uses the ``cl100k_base`` tiktoken encoding to measure token counts.
    Each chunk inherits the metadata of the :class:`RawContent` it was
    derived from.
    """
    chunks: list[Chunk] = []
    for raw in contents:
        text = raw.text
        tokens = _ENCODING.encode(text)
        if len(tokens) == 0:
            continue

        start = 0
        while start < len(tokens):
            end = start + config.chunk_size
            window = tokens[start:end]
            chunk_text = _ENCODING.decode(window)
            chunks.append(
                Chunk(
                    source_id="",
                    kb_id="",
                    content=chunk_text,
                    metadata=dict(raw.metadata),
                )
            )
            # Advance by (chunk_size - overlap) so consecutive windows overlap.
            step = config.chunk_size - config.chunk_overlap
            if step <= 0:
                step = 1  # safety: avoid infinite loop
            start += step

    return chunks


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Ingestor(Protocol):
    """Interface that every source ingestor must satisfy."""

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]: ...

    def chunk(self, contents: list[RawContent], config: ChunkConfig) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# Registry (populated after concrete ingestor imports below)
# ---------------------------------------------------------------------------


def _build_registry() -> dict[SourceType, type[Ingestor]]:
    """Import concrete ingestors and assemble the registry.

    Done inside a function to avoid circular-import issues and so that
    import errors in optional dependencies surface only when actually needed.
    """
    from agent_knowledgebase.ingestors.api_endpoint import ApiEndpointIngestor
    from agent_knowledgebase.ingestors.codebase import CodebaseIngestor
    from agent_knowledgebase.ingestors.directory import DirectoryIngestor
    from agent_knowledgebase.ingestors.file import FileIngestor
    from agent_knowledgebase.ingestors.git_history import GitHistoryIngestor
    from agent_knowledgebase.ingestors.sql_database import SqlDatabaseIngestor
    from agent_knowledgebase.ingestors.website import WebsiteIngestor

    return {
        SourceType.file: FileIngestor,
        SourceType.directory: DirectoryIngestor,
        SourceType.codebase: CodebaseIngestor,
        SourceType.website: WebsiteIngestor,
        SourceType.sql_database: SqlDatabaseIngestor,
        SourceType.git_history: GitHistoryIngestor,
        SourceType.api_endpoint: ApiEndpointIngestor,
    }


INGESTOR_REGISTRY: dict[SourceType, type[Ingestor]] = _build_registry()

__all__ = [
    "ChunkConfig",
    "INGESTOR_REGISTRY",
    "Ingestor",
    "RawContent",
    "token_chunk",
]
