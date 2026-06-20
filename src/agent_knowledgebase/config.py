"""Configuration via environment variables using pydantic-settings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from agent_knowledgebase.config_files import (
    NestedJsonConfigSettingsSource,
    resolve_project_config_path,
    resolve_user_config_path,
)


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
        default_factory=lambda: Path.home() / ".agent-kb" / "saves",
        description="Base directory for knowledgebase storage (AGENT_KB_SAVES_DIR). "
        "Each KB gets its own subdirectory directly under <saves_dir>/<sanitized-name>/. "
        "Defaults to ~/.agent-kb/saves so a fresh install works with no env "
        "vars; resolve_paths() still requires the directory to exist on disk.",
    )

    # --- Vectorstore robustness (Phase A) ---
    chromadb_eager_warm: bool = Field(
        default=True,
        description="Eager-warm chromadb collections at MCP server startup "
        "(AGENT_KB_CHROMADB_EAGER_WARM). When True, all chromadb-backed KB "
        "collections are loaded into memory on server init via a non-blocking "
        "daemon thread, amortising the ~180s HNSW cold-load into startup "
        "time rather than the first tool call. Set to False to disable "
        "(opt-out). spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b",
    )

    # --- Embeddings ---
    embedding_provider: Literal["ollama", "sentence-transformers", "openai"] = Field(
        default="ollama",
        description="Embedding provider (AGENT_KB_EMBEDDING_PROVIDER): "
        "'ollama' (default; uses Ollama's /api/embed at AGENT_KB_EMBED_BASE_URL "
        "with model AGENT_KB_EMBEDDING_MODEL — defaults to qwen3-embedding:8b "
        "against the bundled Ollama sidecar at http://ollama:11434), "
        "'sentence-transformers' (local, offline; bundled in the Docker image), "
        "or 'openai' (POSTs to https://api.openai.com/v1/embeddings using "
        "OPENAI_API_KEY from the environment — rejected loudly when the "
        "env var is unset).",
    )
    embedding_model: str = Field(
        default="qwen3-embedding:8b",
        description="Embedding model name. Default targets Ollama's "
        "qwen3-embedding:8b. Switch to e.g. 'all-MiniLM-L6-v2' for "
        "sentence-transformers or 'text-embedding-3-small' for openai.",
    )
    embed_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the embeddings endpoint. For "
        "embedding_provider = 'ollama' the default is 'http://ollama:11434' "
        "(the '/api/embed' suffix is appended automatically). For "
        "embedding_provider = 'openai' the default is "
        "'https://api.openai.com/v1' — the '/embeddings' suffix is "
        "appended automatically. Ignored by 'sentence-transformers'.",
    )
    embed_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Per-call wall-clock bound in seconds for remote embed requests. "
        "Prevents kb_query from blocking the MCP RPC when the backend is unreachable "
        "or cold-loading a model.",
    )
    embed_max_retries: int = Field(
        default=0,
        ge=0,
        description="Number of retries after a transport failure (timeout / connection "
        "error). Default 0 keeps the wall-clock bounded at one embed_timeout_seconds "
        "window.",
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

    # --- MMR (Maximal Marginal Relevance) tuning — Phase 5 ---
    # B-03: the v0.11.0 release initially shipped two
    # ``query_mmr_lambda_*`` fields and a ``services.query.mmr_lambda_for``
    # helper. Code-review caught that nothing in
    # :class:`QueryOrchestrator` actually wires MMR into the query path,
    # so the fields + helper were dead surface. Both were removed in the
    # follow-up. The empirical Jaccard@10 ≥ 0.95 parity result between
    # fastembed-int8 and ST-fp32 (see
    # ``tests/test_fastembed_recall_parity.py``) is the live safety net
    # for the fastembed swap; MMR re-tuning is deferred until an actual
    # MMR rerank pass lands. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b

    # --- Ingest ---
    ingest_excluded_dirs: list[str] | str = Field(
        default_factory=lambda: [
            "__pycache__",
            "node_modules",
            ".git",
            ".venv",
            ".mypy_cache",
            ".pytest_cache",
            "dist",
            "build",
            "venv",
            ".tox",
            ".ruff_cache",
            ".eggs",
            ".idea",
            ".vscode",
            ".hg",
            ".svn",
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
            raise FileNotFoundError(f"AGENT_KB_SAVES_DIR does not exist: {resolved.saves_dir}")
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        user_source = NestedJsonConfigSettingsSource(settings_cls, path=resolve_user_config_path())
        project_source = NestedJsonConfigSettingsSource(
            settings_cls, path=resolve_project_config_path()
        )
        # Left-to-right is highest-to-lowest priority in pydantic-settings.
        # Order yields: init > env > dotenv > project JSON > user JSON > defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            project_source,
            user_source,
            file_secret_settings,
        )
