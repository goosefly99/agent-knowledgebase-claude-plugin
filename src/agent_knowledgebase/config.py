"""Configuration via environment variables using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Agent Knowledgebase configuration.

    All settings can be overridden via environment variables prefixed with ``AGENT_KB_``.
    """

    model_config = {"env_prefix": "AGENT_KB_"}

    # --- Storage ---
    db_path: Path = Field(
        default=Path("~/.agent-kb/knowledgebase.db"),
        description="SQLite database path",
    )

    # --- Vector store ---
    vectorstore: Literal["chromadb", "pinecone"] = Field(
        default="chromadb",
        description="Vector store backend",
    )
    chroma_path: Path = Field(
        default=Path("~/.agent-kb/chroma"),
        description="ChromaDB persistence directory",
    )

    # --- Pinecone ---
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key",
    )
    pinecone_index: Optional[str] = Field(
        default=None,
        description="Pinecone index name",
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Pinecone environment",
    )

    # --- Embeddings ---
    embedding_provider: Literal["sentence-transformers", "openai"] = Field(
        default="sentence-transformers",
        description="Embedding provider",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key",
    )

    # --- Chunking ---
    chunk_size: int = Field(
        default=512,
        description="Default chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Chunk overlap in tokens",
    )

    # --- Export ---
    export_path: Optional[Path] = Field(
        default=None,
        description="Markdown export directory",
    )

    def resolve_paths(self) -> Settings:
        """Return a copy with all ``~`` paths expanded to absolute paths."""
        updates: dict[str, Path] = {}
        for field_name in ("db_path", "chroma_path", "export_path"):
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = Path(value).expanduser().resolve()
        return self.model_copy(update=updates)
