"""Configuration via environment variables using pydantic-settings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
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

    # --- Retrieval backend (Phase 2 redesign) ---
    kb_backend: Literal["chromadb", "markdown", "lightrag", "textvec"] = Field(
        default="chromadb",
        validation_alias=AliasChoices(
            "kb_backend",
            "AGENT_KB_BACKEND",
            "AGENT_KB_KB_BACKEND",
        ),
        description="Retrieval backend selector (AGENT_KB_BACKEND, also "
        "accepted as AGENT_KB_KB_BACKEND for env_prefix consistency). "
        "Picks the high-level retrieval strategy that wraps "
        "ingest/query/search/delete: 'chromadb' (default; vector + FTS via "
        "the existing chromadb pipeline), 'markdown' (Phase 3, opt-in "
        "Karpathy-style wiki backend), 'lightrag' (Phase 6, deferred), or "
        "'textvec' (Phase B, opt-in; SQLite FTS5+BM25 over the existing "
        "chunks table — zero new external dependencies). "
        "Disambiguation: this is NOT the same as the 'vectorstore' field. "
        "'kb_backend' selects the retrieval-strategy abstraction "
        "(RetrieverBackend); 'vectorstore' selects the chromadb-internal "
        "vector-store provider in {chromadb, pinecone}, used only by the "
        "ChromadbBackend. The two compose: kb_backend='chromadb' + "
        "vectorstore='pinecone' would route through ChromadbBackend with a "
        "Pinecone-backed VectorStore. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b",
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

    # --- TextvecBackend tuning (Phase B, v2.2) ---
    # These fields are consumed by TextvecBackend (AGENT_KB_BACKEND=textvec).
    # They have no effect when kb_backend is 'chromadb', 'markdown', or
    # 'lightrag'. bm25_k1 / bm25_b are stored for documentation and future
    # use; SQLite FTS5's BM25 parameters are currently configured via the
    # tokenize= option and rank order, not via explicit k1/b tuning.
    # spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
    fts5_tokenizer: str = Field(
        default="porter unicode61",
        description="FTS5 tokenizer directive (AGENT_KB_FTS5_TOKENIZER). "
        "Default 'porter unicode61' enables English stemming and Unicode "
        "normalisation. See SQLite FTS5 docs for supported tokenizer names.",
    )
    bm25_k1: float = Field(
        default=1.2,
        gt=0.0,
        description="BM25 term-frequency saturation parameter k1 "
        "(AGENT_KB_BM25_K1). Stored for documentation; SQLite FTS5 uses "
        "its own BM25 implementation with fixed k1.",
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 document-length normalisation parameter b "
        "(AGENT_KB_BM25_B). Stored for documentation; SQLite FTS5 uses "
        "its own BM25 implementation with fixed b.",
    )
    lexical_min_token_len: int = Field(
        default=2,
        ge=1,
        description="Minimum token length for FTS5 queries "
        "(AGENT_KB_LEXICAL_MIN_TOKEN_LEN). Tokens shorter than this are "
        "stripped from search queries to reduce noise. Default 2.",
    )

    # --- Per-KB retrieval backend overrides (Phase 4 redesign) ---
    kb_backend_per_kb: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "kb_backend_per_kb",
            "AGENT_KB_BACKEND_PER_KB",
            "AGENT_KB_KB_BACKEND_PER_KB",
        ),
        description="Per-knowledgebase backend overrides (Phase 4). "
        "Comma-separated ``kb_id=backend`` pairs that override "
        "``kb_backend`` for the listed KB ids. Example env: "
        "``AGENT_KB_BACKEND_PER_KB='kb1=markdown,kb2=chromadb'``. "
        "Resolution precedence at backend-selection time is: "
        "(1) <saves_dir>/<kb-name>/.migrated_to sentinel (written by "
        "kb_migrate after a successful migration), "
        "(2) this kb_backend_per_kb mapping, "
        "(3) the global ``Settings.kb_backend``. "
        "Values must be one of {'chromadb', 'markdown', 'lightrag', 'textvec'}; "
        "invalid backend names are rejected at validation time. "
        "spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b",
    )

    @field_validator("kb_backend_per_kb", mode="before")
    @classmethod
    def _parse_kb_backend_per_kb(cls, value: object) -> object:
        """Accept either a dict or a comma-separated ``kb_id=backend`` string.

        pydantic-settings would normally try to JSON-decode an env value
        for a ``dict`` field; declaring this validator with
        ``mode='before'`` lets us accept the friendlier
        ``kb1=markdown,kb2=chromadb`` form that the spec calls for
        (parallel to the existing ``ingest_excluded_dirs`` parser).
        """
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            # Best-effort JSON first so users can still pass strict JSON
            # if they prefer; fall through to the comma-separated form
            # on parse failure.
            if text.startswith("{"):
                import json as _json

                try:
                    parsed = _json.loads(text)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
            out: dict[str, str] = {}
            for pair in text.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if "=" not in pair:
                    raise ValueError(
                        f"AGENT_KB_BACKEND_PER_KB entry {pair!r} is not "
                        f"a valid 'kb_id=backend' pair"
                    )
                kb_id, backend = pair.split("=", 1)
                kb_id = kb_id.strip()
                backend = backend.strip()
                if not kb_id or not backend:
                    raise ValueError(
                        f"AGENT_KB_BACKEND_PER_KB entry {pair!r} has an empty kb_id or backend"
                    )
                out[kb_id] = backend
            return out
        return value

    @field_validator("kb_backend_per_kb", mode="after")
    @classmethod
    def _validate_kb_backend_per_kb_values(cls, value: dict[str, str]) -> dict[str, str]:
        valid = {"chromadb", "markdown", "lightrag", "textvec"}
        for kb_id, backend in value.items():
            if backend not in valid:
                raise ValueError(
                    f"AGENT_KB_BACKEND_PER_KB[{kb_id!r}]={backend!r} is "
                    f"not a valid backend; expected one of {sorted(valid)}"
                )
        return value

    # --- Vector store (chromadb-internal provider selection) ---
    vectorstore: Literal["chromadb", "pinecone"] = Field(
        default="chromadb",
        description="Vector store provider used INSIDE the ChromadbBackend "
        "to back the per-KB vector collection: 'chromadb' (local "
        "PersistentClient, default) or 'pinecone' (remote, requires "
        "AGENT_KB_PINECONE_* credentials). This is NOT the same setting "
        "as 'kb_backend' — see kb_backend's docstring for the layering.",
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
    embedding_provider: Literal["ollama", "sentence-transformers", "remote", "fastembed"] = Field(
        default="remote",
        description="Embedding provider: 'remote' (default since v0.11.0 — "
        "HTTP endpoint speaking the OpenAI-compatible /v1/embeddings JSON "
        "contract — e.g. OpenAI proper, vLLM, LocalAI, or Ollama's /v1 "
        "surface), 'ollama' (native /api/embed, no API key), "
        "'sentence-transformers' (local, offline; install via "
        "'pip install agent-knowledgebase[embed-local-st]'), or "
        "'fastembed' (local ONNX runtime; install via "
        "'pip install agent-knowledgebase[embed-local-onnx]'). "
        "The Phase 5 default-flip from 'ollama' to 'remote' relies on "
        "the Phase 4 per-chunk provider snapshot — existing v0.6.0 KBs "
        "ingested under 'ollama' stay queryable post-flip because the "
        "snapshot rebuilds the original embedder via "
        "create_embedder_for_model(model, provider=..., base_url=...). "
        "spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name. Default since v0.11.0 targets "
        "OpenAI's 'text-embedding-3-small' (1536-dim). Switch to e.g. "
        "'qwen3-embedding:8b' when embedding_provider = 'ollama', "
        "'all-MiniLM-L6-v2' for sentence-transformers, or "
        "'BAAI/bge-small-en-v1.5' for fastembed. spec_id: "
        "70ab2170-381a-4657-bcd1-28a40c6f369b",
    )
    embed_api_key: Optional[str] = Field(
        default=None,
        description="API key sent as Bearer auth to the remote embeddings "
        "endpoint. Required for embedding_provider = 'remote'; ignored by "
        "the 'ollama' and 'sentence-transformers' providers.",
    )
    embed_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the embeddings endpoint. For "
        "embedding_provider = 'ollama' the default is 'http://127.0.0.1:11434' "
        "(the '/api/embed' suffix is appended automatically). For "
        "embedding_provider = 'remote' this must be the full OpenAI-compatible "
        "base (e.g. 'http://localhost:11434/v1') — the '/embeddings' suffix is "
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

    @model_validator(mode="after")
    def _validate_kb_backend_vectorstore_consistency(self) -> Settings:
        """``vectorstore`` is only meaningful when ``kb_backend='chromadb'``.

        The ``vectorstore`` field selects the chromadb-internal vector-store
        provider (``chromadb`` PersistentClient vs. ``pinecone``). It is
        consumed exclusively by ``ChromadbBackend`` / ``services/vectorstore.py``
        — the ``markdown`` and ``lightrag`` backends don't read it. Accepting
        e.g. ``kb_backend='markdown' + vectorstore='pinecone'`` therefore
        encodes a meaningless combination that almost certainly reflects a
        configuration error. Reject it loudly so Phase 3 doesn't silently
        ignore the operator's intent.

        The default ``chromadb`` value of ``vectorstore`` is treated as
        benign even when ``kb_backend != 'chromadb'`` (it's the field
        default and may simply be unset).
        """
        if (
            self.kb_backend in {"markdown", "lightrag", "textvec"}
            and self.vectorstore != "chromadb"
        ):
            msg = (
                f"vectorstore={self.vectorstore!r} is meaningless when "
                f"kb_backend={self.kb_backend!r}; only set vectorstore "
                f"when kb_backend='chromadb'"
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
