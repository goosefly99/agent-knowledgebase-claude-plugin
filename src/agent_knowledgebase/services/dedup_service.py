"""Deduplication policy enum and resolver for kb_ingest / kb_ingest_batch."""

from __future__ import annotations

import enum
from typing import Optional, Tuple

from agent_knowledgebase.database import Database
from agent_knowledgebase.models import Source


class DedupPolicy(str, enum.Enum):
    skip = "skip"
    replace = "replace"
    force_add = "force_add"


DEFAULT_DEDUP_POLICY: DedupPolicy = DedupPolicy.skip

# DedupAction is one of:
#   ("insert", None)          — no existing match; proceed normally
#   ("skip", existing_source) — match found; caller returns existing unchanged
#   ("replace", existing_source) — match found; caller removes old, then inserts
DedupAction = Tuple[str, Optional[Source]]


def resolve_dedup_action(
    db: Database,
    kb_id: str,
    dedup_key: Optional[str],
    policy: DedupPolicy,
) -> DedupAction:
    """Return the action the caller should take before inserting a new Source.

    When *dedup_key* is None or empty no lookup is performed and the caller
    always proceeds with a normal insert.
    """
    if not dedup_key:
        return ("insert", None)

    existing = db.find_source_by_dedup_key(kb_id, dedup_key)
    if existing is None:
        return ("insert", None)

    if policy == DedupPolicy.skip:
        return ("skip", existing)
    if policy == DedupPolicy.replace:
        return ("replace", existing)
    # force_add — ignore the existing row
    return ("insert", None)
