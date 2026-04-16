"""Pydantic data models for agent-knowledgebase entities."""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, enum.Enum):
    """Kind of data source that can be ingested."""

    file = "file"
    directory = "directory"
    codebase = "codebase"
    website = "website"
    sql_database = "sql_database"
    git_history = "git_history"
    api_endpoint = "api_endpoint"


class SourceStatus(str, enum.Enum):
    """Lifecycle status of a source."""

    pending = "pending"
    ingesting = "ingesting"
    ingested = "ingested"
    failed = "failed"


class PageType(str, enum.Enum):
    """Category of wiki page."""

    entity = "entity"
    concept = "concept"
    summary = "summary"
    index = "index"
    comparison = "comparison"
    synthesis = "synthesis"


class PipelinePhase(str, enum.Enum):
    """Named phase within the ingestion pipeline."""

    initialize = "initialize"
    read_source = "read_source"
    chunk = "chunk"
    embed = "embed"
    integrate_wiki = "integrate_wiki"
    finalize = "finalize"


class RunStatus(str, enum.Enum):
    """Execution status of a pipeline run."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC time (used as a default factory)."""
    return datetime.now(UTC)


def _uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid4())


class Knowledgebase(BaseModel):
    """A top-level knowledgebase container."""

    id: str = Field(default_factory=_uuid)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    source_count: int = Field(default=0, description="Computed — not stored in DB")
    page_count: int = Field(default=0, description="Computed — not stored in DB")
    config: dict = Field(default_factory=dict)


class Source(BaseModel):
    """A data source attached to a knowledgebase."""

    id: str = Field(default_factory=_uuid)
    kb_id: str
    source_type: SourceType
    uri: str
    metadata: dict = Field(default_factory=dict)
    ingested_at: Optional[datetime] = None
    chunk_count: int = 0
    status: SourceStatus = SourceStatus.pending
    dedup_key: Optional[str] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_id(self) -> str:
        return self.id


class WikiPage(BaseModel):
    """A wiki page generated from ingested sources."""

    id: str = Field(default_factory=_uuid)
    kb_id: str
    title: str
    content: str
    page_type: PageType
    source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    inbound_links: list[str] = Field(default_factory=list)
    outbound_links: list[str] = Field(default_factory=list)
    source_type: Optional[str] = Field(
        default=None,
        description=(
            "Denormalized from page.source_ids[0] (first entry in list order, not "
            "chronological). Populated by KnowledgebaseService.list_pages only — "
            "get_page, direct model construction, and direct DB reads leave this None. "
            "Not persisted on the wiki_pages table."
        ),
    )
    uri: Optional[str] = Field(
        default=None,
        description=(
            "Denormalized from page.source_ids[0] (first entry in list order, not "
            "chronological). Populated by KnowledgebaseService.list_pages only — "
            "get_page, direct model construction, and direct DB reads leave this None. "
            "Not persisted on the wiki_pages table."
        ),
    )
    dedup_key: Optional[str] = Field(
        default=None,
        description=(
            "Denormalized from page.source_ids[0] (first entry in list order, not "
            "chronological). Populated by KnowledgebaseService.list_pages only — "
            "get_page, direct model construction, and direct DB reads leave this None. "
            "Not persisted on the wiki_pages table."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def page_id(self) -> str:
        return self.id


class Chunk(BaseModel):
    """A text chunk extracted from a source."""

    id: str = Field(default_factory=_uuid)
    source_id: str
    kb_id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    embedding_id: Optional[str] = None


class PipelineRun(BaseModel):
    """A single pipeline execution record."""

    id: str = Field(default_factory=_uuid)
    kb_id: str
    source_id: Optional[str] = None
    phase: PipelinePhase
    status: RunStatus = RunStatus.pending
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
