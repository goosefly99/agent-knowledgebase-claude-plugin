"""User- and project-level JSON config file support for agent-knowledgebase.

This module is independent of :mod:`agent_knowledgebase.config` to avoid
circular imports — :class:`Settings` imports from here, not vice-versa.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

logger = logging.getLogger("agent_knowledgebase.config")

# Module-level overrides consulted by resolve_*_config_path().  Used by
# load_settings() to perform a dry-run against a candidate file without
# mutating the real on-disk state.  Public API is the helper functions —
# tests should prefer those.
_USER_CONFIG_PATH_OVERRIDE: Path | None = None
_PROJECT_CONFIG_PATH_OVERRIDE: Path | None = None


def resolve_user_config_path() -> Path:
    """Return the absolute path to the user-level config file.

    Discovery order:
      1. Module-level override set by :func:`load_settings` (test + dry-run aid).
      2. ``$AGENT_KB_USER_CONFIG`` environment variable.
      3. ``~/.agent-kb/config.json`` default.
    """
    if _USER_CONFIG_PATH_OVERRIDE is not None:
        return _USER_CONFIG_PATH_OVERRIDE
    env_value = os.environ.get("AGENT_KB_USER_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / ".agent-kb" / "config.json"


def resolve_project_config_path() -> Path:
    """Return the absolute path to the project-level config file.

    Discovery order:
      1. Module-level override set by :func:`load_settings`.
      2. ``$AGENT_KB_PROJECT_CONFIG`` explicit file path.
      3. ``<project_root>/.agent-kb/config.json`` where <project_root> is
         derived from, in order: ``$AGENT_KB_PROJECT_DIR``,
         ``$CLAUDE_PROJECT_DIR``, ``$PWD``, then :func:`os.getcwd`.
    """
    if _PROJECT_CONFIG_PATH_OVERRIDE is not None:
        return _PROJECT_CONFIG_PATH_OVERRIDE
    env_file = os.environ.get("AGENT_KB_PROJECT_CONFIG")
    if env_file:
        return Path(env_file).expanduser()

    for env_name in ("AGENT_KB_PROJECT_DIR", "CLAUDE_PROJECT_DIR", "PWD"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser() / ".agent-kb" / "config.json"

    return Path(os.getcwd()) / ".agent-kb" / "config.json"


# Dotted JSON paths → flat Settings field names.  See design spec §5.2.
DOT_TO_FLAT: dict[str, str] = {
    "kb_backend": "kb_backend",
    "vectorstore": "vectorstore",
    "embedding.provider": "embedding_provider",
    "embedding.model": "embedding_model",
    "embedding.base_url": "embed_base_url",
    "embedding.timeout_seconds": "embed_timeout_seconds",
    "embedding.max_retries": "embed_max_retries",
    "pinecone.index": "pinecone_index",
    "pinecone.environment": "pinecone_environment",
    "export_path": "export_path",
    "chunk.size": "chunk_size",
    "chunk.overlap": "chunk_overlap",
    "chunk.token_encoding": "chunk_token_encoding",
    "query.default_top_k": "query_default_top_k",
    "query.hybrid.vector_weight": "query_hybrid_vector_weight",
    "query.hybrid.fts_weight": "query_hybrid_fts_weight",
    "query.hybrid.fetch_multiplier": "query_hybrid_fetch_multiplier",
    "ingest.excluded_dirs": "ingest_excluded_dirs",
}

# Keys (at any depth or any flat form) that must never appear in a JSON config.
# The JSON source rejects them loudly because they belong in env (secrets or
# required+validated file paths).
FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "embed_api_key", "pinecone_api_key", "saves_dir", "api_key",
})


def _flatten_dotted(raw: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Walk a nested dict, return a flat dict keyed by dotted paths.

    ``{"embedding": {"model": "x"}}`` → ``{"embedding.model": "x"}``.
    """
    out: dict[str, Any] = {}
    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_dotted(value, prefix=path))
        else:
            out[path] = value
    return out


class NestedJsonConfigSettingsSource(PydanticBaseSettingsSource):
    """Pydantic-settings source that reads a single nested JSON config file.

    Responsibilities:
      * Return ``{}`` with a warning if the file is missing or empty ``{}``.
      * Raise on malformed JSON, unknown keys, forbidden (secret) keys.
      * Flatten dotted paths into :class:`Settings` flat field names.
    """

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    # pydantic-settings calls __call__ to get the dict of values.
    def __call__(self) -> dict[str, Any]:
        if not self._path.exists():
            logger.warning(
                "agent-knowledgebase: config file not found at %s — using defaults + env.",
                self._path,
            )
            return {}

        text = self._path.read_text(encoding="utf-8").strip()
        if not text or text == "{}":
            logger.warning(
                "agent-knowledgebase: config file %s is empty — using defaults + env.",
                self._path,
            )
            return {}

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"agent-knowledgebase: failed to parse JSON in {self._path}: {exc}"
            raise ValueError(msg) from exc

        if not isinstance(raw, dict):
            msg = f"agent-knowledgebase: top-level of {self._path} must be a JSON object."
            raise ValueError(msg)

        flat = _flatten_dotted(raw)

        # 1. Reject forbidden keys anywhere in the tree.
        for dotted in flat:
            # Check leaf name *and* any segment matching a forbidden name.
            for segment in dotted.split("."):
                if segment in FORBIDDEN_KEYS:
                    flat_equiv = dotted.replace(".", "_")
                    raise ValueError(
                        f"agent-knowledgebase: config file {self._path} contains forbidden key "
                        f"'{dotted}' (flat equivalent: '{flat_equiv}', segment '{segment}'). "
                        f"Secrets and saves_dir must be set via environment variable, never in "
                        f"a JSON config file."
                    )

        # 2. Reject unknown dotted keys.
        unknown = [k for k in flat if k not in DOT_TO_FLAT]
        if unknown:
            valid_list = ", ".join(sorted(DOT_TO_FLAT))
            raise ValueError(
                f"agent-knowledgebase: config file {self._path} contains unknown keys: "
                f"{unknown}. Valid keys are: {valid_list}"
            )

        # 3. Remap dotted → flat and return.
        return {DOT_TO_FLAT[k]: v for k, v in flat.items()}

    # Required by pydantic-settings but unused for whole-file sources.
    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        return None, field_name, False


if TYPE_CHECKING:
    from agent_knowledgebase.config import Settings


@contextmanager
def _path_overrides(
    user_path: Path | None, project_path: Path | None
) -> Iterator[None]:
    """Temporarily patch module-level overrides for :func:`load_settings`."""
    global _USER_CONFIG_PATH_OVERRIDE, _PROJECT_CONFIG_PATH_OVERRIDE
    prev_user = _USER_CONFIG_PATH_OVERRIDE
    prev_project = _PROJECT_CONFIG_PATH_OVERRIDE
    if user_path is not None:
        _USER_CONFIG_PATH_OVERRIDE = user_path
    if project_path is not None:
        _PROJECT_CONFIG_PATH_OVERRIDE = project_path
    try:
        yield
    finally:
        _USER_CONFIG_PATH_OVERRIDE = prev_user
        _PROJECT_CONFIG_PATH_OVERRIDE = prev_project


def load_settings(
    user_path: Path | None = None, project_path: Path | None = None
) -> "Settings":
    """Construct a fresh :class:`Settings` with optional path overrides.

    Used at server startup (no args → normal discovery) and by
    :func:`kb_config_set` to dry-run a candidate file before committing.
    """
    # Local import avoids a circular dependency at module load.
    from agent_knowledgebase.config import Settings

    with _path_overrides(user_path, project_path):
        return Settings()


def _nested_set(raw: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``raw`` with ``dotted_key`` set to ``value``.

    Non-destructive for sibling keys at the same depth: ``{"a": {"b": 1}}``
    with ``set("a.c", 2)`` → ``{"a": {"b": 1, "c": 2}}``.

    If an intermediate segment exists in ``raw`` but is not a dict (e.g., a
    scalar left over from a hand-edited config), it is silently replaced
    with an empty dict so the nested assignment can proceed. Callers that
    care about such malformed input should validate the returned dict via
    ``load_settings`` — the normal Task 10 flow already does this.
    """
    out = copy.deepcopy(raw)
    segments = dotted_key.split(".")
    cursor = out
    for segment in segments[:-1]:
        if segment not in cursor or not isinstance(cursor[segment], dict):
            cursor[segment] = {}
        cursor = cursor[segment]
    cursor[segments[-1]] = value
    return out


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically (same-dir tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_candidate_and_validate(
    target_path: Path, dotted_key: str, value: Any, *, scope: str
) -> None:
    """Write a candidate update to ``<target>.tmp``, validate via
    :func:`load_settings`, then atomically replace the real file on success.

    On any validation failure, removes the tmp file and re-raises.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if target_path.exists():
        text = target_path.read_text(encoding="utf-8")
        if text.strip():
            current = json.loads(text)

    updated = _nested_set(current, dotted_key, value)
    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp.write_text(json.dumps(updated, indent=2), encoding="utf-8")

    try:
        if scope == "user":
            load_settings(user_path=tmp)
        else:
            load_settings(project_path=tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    os.replace(tmp, target_path)
