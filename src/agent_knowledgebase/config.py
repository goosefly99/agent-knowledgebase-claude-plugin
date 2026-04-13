"""Configuration via environment variables using pydantic-settings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def sanitize_kb_dir_name(name: str) -> str:
    """Convert a knowledgebase name to a filesystem-safe directory name.

    >>> sanitize_kb_dir_name("My Research KB!")
    'my-research-kb'
    >>> sanitize_kb_dir_name("  hello   world  ")
    'hello-world'
    """
    safe = re.sub(r"[^\w\s-]", "", name.lower())
    safe = re.sub(r"[\s_]+", "-", safe)
    safe = safe.strip("-")
    return safe or "unnamed"


class Settings(BaseSettings):
    """Agent Knowledgebase configuration.

    All settings can be overridden via environment variables prefixed with ``AGENT_KB_``.
    """

    model_config = SettingsConfigDict(env_prefix="AGENT_KB_")

    # --- Storage ---
    saves_dir: Path = Field(
        description="Base directory for knowledgebase storage (AGENT_KB_SAVES_DIR). "
        "Each KB gets its own subdirectory directly under <saves_dir>/<sanitized-name>/. "
        "This environment variable is required and the directory must exist.",
    )

    # --- Vector store ---
    vectorstore: Literal["chromadb", "pinecone"] = Field(
        default="chromadb",
        description="Vector store backend",
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
        gt=0,
        description="Default chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Chunk overlap in tokens",
    )
    chunk_token_encoding: str = Field(
        default="cl100k_base",
        description="tiktoken encoding name used for token-based chunking",
    )

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            msg = f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_hybrid_weights(self) -> Settings:
        total = self.query_hybrid_vector_weight + self.query_hybrid_fts_weight
        if abs(total - 1.0) > 1e-6:
            msg = (
                f"query_hybrid_vector_weight ({self.query_hybrid_vector_weight}) + "
                f"query_hybrid_fts_weight ({self.query_hybrid_fts_weight}) must sum to 1.0, "
                f"got {total}"
            )
            raise ValueError(msg)
        return self

    # --- Query ---
    query_default_top_k: int = Field(
        default=10,
        gt=0,
        description="Default number of results returned by kb_query and kb_search",
    )
    query_hybrid_vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight of vector scores in hybrid search (must sum to 1.0 with fts_weight)",
    )
    query_hybrid_fts_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of full-text-search scores in hybrid search",
    )
    query_hybrid_fetch_multiplier: int = Field(
        default=2,
        ge=1,
        description="Over-fetch multiplier used to merge hybrid results before trimming to top_k",
    )

    # --- Ingest ---
    ingest_excluded_dirs: list[str] | str = Field(
        default_factory=lambda: [
            "__pycache__", "node_modules", ".git", ".venv",
            ".mypy_cache", ".pytest_cache", "dist", "build",
        ],
        description="Directory names to skip during recursive ingestion. "
        "This is a full replacement of the default list when set.",
    )

    @field_validator("ingest_excluded_dirs", mode="before")
    @classmethod
    def _parse_excluded_dirs(cls, value: object) -> object:
        # pydantic-settings treats pure list[str] as complex and JSON-decodes the env string,
        # rejecting non-JSON input.  Declaring the type as list[str] | str tells pydantic-settings
        # to allow parse failure and pass the raw string through to this validator, which then
        # splits on commas.  At runtime the return value is always list[str].
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            return [p for p in parts if p]
        return value

    # --- Export ---
    export_path: Optional[Path] = Field(
        default=None,
        description="Markdown export directory",
    )

    def resolve_paths(self) -> Settings:
        """Return a copy with all ``~`` paths expanded to absolute paths.

        Raises
        ------
        NotADirectoryError
            If the resolved *saves_dir* path exists but is not a directory.
        FileNotFoundError
            If the resolved *saves_dir* does not exist on disk.
        """
        updates: dict[str, Path] = {}
        for field_name in ("saves_dir", "export_path"):
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = Path(value).expanduser().resolve()
        resolved = self.model_copy(update=updates)
        if not resolved.saves_dir.exists():
            raise FileNotFoundError(
                f"AGENT_KB_SAVES_DIR does not exist: {resolved.saves_dir}"
            )
        if not resolved.saves_dir.is_dir():
            raise NotADirectoryError(
                f"AGENT_KB_SAVES_DIR exists but is not a directory: {resolved.saves_dir}"
            )
        return resolved

    # --- Per-KB path helpers ---

    @property
    def knowledgebases_dir(self) -> Path:
        """Base directory under which each KB gets its own subdirectory."""
        return self.saves_dir

    def kb_data_dir(self, dir_name: str) -> Path:
        """Data directory for a single knowledgebase."""
        return self.saves_dir / dir_name

    def kb_db_path(self, dir_name: str) -> Path:
        """SQLite database path for a single knowledgebase."""
        return self.kb_data_dir(dir_name) / "knowledgebase.db"

    def kb_chroma_path(self, dir_name: str) -> Path:
        """ChromaDB persistence directory for a single knowledgebase."""
        return self.kb_data_dir(dir_name) / "chroma"
