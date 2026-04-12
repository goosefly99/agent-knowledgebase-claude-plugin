"""IngestionOrchestrator -- dispatch to the correct ingestor for a source type."""

from __future__ import annotations

from agent_knowledgebase.config import Settings
from agent_knowledgebase.ingestors import INGESTOR_REGISTRY, ChunkConfig
from agent_knowledgebase.models import Chunk, SourceType


class IngestionOrchestrator:
    """High-level service that selects the right ingestor and runs it.

    Usage::

        orchestrator = IngestionOrchestrator(settings)
        chunks = orchestrator.ingest(SourceType.file, "/path/to/file.md")
    """

    def __init__(self, config: Settings) -> None:
        self._config = config

    def ingest(
        self,
        source_type: SourceType,
        uri: str,
        metadata: dict | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> list[Chunk]:
        """Look up the ingestor, read the source, chunk content, and return chunks."""
        ingestor_cls = INGESTOR_REGISTRY.get(source_type)
        if ingestor_cls is None:
            raise ValueError(f"No ingestor registered for source type: {source_type}")

        ingestor = ingestor_cls()
        contents = ingestor.read(uri, metadata)

        effective_config = chunk_config or ChunkConfig(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )

        return ingestor.chunk(contents, effective_config)
