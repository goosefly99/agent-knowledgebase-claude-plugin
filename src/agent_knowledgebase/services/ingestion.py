"""IngestionOrchestrator -- dispatch to the correct ingestor for a source type."""

from __future__ import annotations

from agent_knowledgebase.config import Settings
from agent_knowledgebase.ingestors import INGESTOR_REGISTRY, ChunkConfig
from agent_knowledgebase.ingestors.codebase import CodebaseIngestor
from agent_knowledgebase.ingestors.directory import DirectoryIngestor, excluded_dirs_for
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

        if ingestor_cls is DirectoryIngestor:
            ingestor = DirectoryIngestor(excluded_dirs=excluded_dirs_for(self._config))
        elif ingestor_cls is CodebaseIngestor:
            ingestor = CodebaseIngestor(excluded_dirs=excluded_dirs_for(self._config))
        else:
            ingestor = ingestor_cls()
        contents = ingestor.read(uri, metadata)

        effective_config = chunk_config or ChunkConfig.from_settings(self._config)

        return ingestor.chunk(contents, effective_config)
