"""Phase 4 — per-KB backend routing tests.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[4].tasks[3] — AGENT_KB_BACKEND_PER_KB
        env: comma-separated kb_id=backend pairs override global
        ``Settings.kb_backend`` per KB.

Test surface
------------

* ``Settings.kb_backend_per_kb`` parses both the comma-separated env
  form and the JSON object form.
* ``backends.resolve_backend_name`` honours precedence: sentinel >
  per-kb mapping > global default.
* ``KnowledgebaseService._backend_for(kb_id)`` returns the right
  backend per kb_id.
* Invalid backend names in the per-KB mapping are rejected at
  Settings validation time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.backends import (
    MIGRATED_SENTINEL_FILENAME,
    resolve_backend_name,
)
from agent_knowledgebase.config import Settings


# ---------------------------------------------------------------------------
# Settings parser
# ---------------------------------------------------------------------------


def test_kb_backend_per_kb_parses_comma_separated_pairs(
    tmp_path: Path,
) -> None:
    """Env-style ``kb1=markdown,kb2=chromadb`` parses into a dict."""
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        kb_backend_per_kb="kb1=markdown,kb2=chromadb",
    ).resolve_paths()
    assert settings.kb_backend_per_kb == {
        "kb1": "markdown",
        "kb2": "chromadb",
    }


def test_kb_backend_per_kb_accepts_dict_directly(tmp_path: Path) -> None:
    """A dict passed in init still works (programmatic config)."""
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        kb_backend_per_kb={"kb1": "markdown", "kb2": "chromadb"},
    ).resolve_paths()
    assert settings.kb_backend_per_kb == {
        "kb1": "markdown",
        "kb2": "chromadb",
    }


def test_kb_backend_per_kb_empty_string_is_empty_dict(tmp_path: Path) -> None:
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(saves_dir=saves, kb_backend_per_kb="").resolve_paths()
    assert settings.kb_backend_per_kb == {}


def test_kb_backend_per_kb_rejects_invalid_backend_name(
    tmp_path: Path,
) -> None:
    """A bogus backend name in the mapping fails validation up-front."""
    saves = tmp_path / "saves"
    saves.mkdir()
    with pytest.raises(Exception, match="not a valid backend"):
        Settings(
            saves_dir=saves,
            kb_backend_per_kb={"kb1": "bogus"},
        )


def test_kb_backend_per_kb_rejects_malformed_pair(tmp_path: Path) -> None:
    """A pair without ``=`` is rejected."""
    saves = tmp_path / "saves"
    saves.mkdir()
    with pytest.raises(Exception, match="not a valid 'kb_id=backend' pair"):
        Settings(
            saves_dir=saves,
            kb_backend_per_kb="kb1markdown",
        )


# ---------------------------------------------------------------------------
# resolve_backend_name precedence
# ---------------------------------------------------------------------------


def test_resolve_backend_name_uses_global_default_when_no_overrides(
    tmp_path: Path,
) -> None:
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(saves_dir=saves, kb_backend="chromadb").resolve_paths()
    assert resolve_backend_name(settings, kb_id="some-kb") == "chromadb"


def test_resolve_backend_name_uses_per_kb_mapping_over_global(
    tmp_path: Path,
) -> None:
    """The per-KB mapping wins over the global default."""
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        kb_backend="chromadb",
        kb_backend_per_kb={"kb-md": "markdown", "kb-cdb": "chromadb"},
    ).resolve_paths()
    assert resolve_backend_name(settings, kb_id="kb-md") == "markdown"
    assert resolve_backend_name(settings, kb_id="kb-cdb") == "chromadb"
    # Unmapped kb_ids still fall through to the global default.
    assert resolve_backend_name(settings, kb_id="kb-other") == "chromadb"


def test_resolve_backend_name_sentinel_overrides_per_kb_mapping(
    tmp_path: Path,
) -> None:
    """The ``.migrated_to`` sentinel beats the per-KB mapping."""
    saves = tmp_path / "saves"
    saves.mkdir()
    # Layout: <saves>/kb-x/.migrated_to=markdown
    kb_dir = saves / "kb-x"
    kb_dir.mkdir()
    (kb_dir / MIGRATED_SENTINEL_FILENAME).write_text("markdown", encoding="utf-8")

    settings = Settings(
        saves_dir=saves,
        kb_backend="chromadb",
        kb_backend_per_kb={"kb-x": "chromadb"},
    ).resolve_paths()
    # Sentinel says markdown, mapping says chromadb -> sentinel wins.
    assert resolve_backend_name(settings, kb_id="kb-x") == "markdown"


def test_resolve_backend_name_invalid_sentinel_value_is_ignored(
    tmp_path: Path,
) -> None:
    """An unrecognized sentinel value falls through to the next layer."""
    saves = tmp_path / "saves"
    saves.mkdir()
    kb_dir = saves / "kb-y"
    kb_dir.mkdir()
    (kb_dir / MIGRATED_SENTINEL_FILENAME).write_text("ohno", encoding="utf-8")

    settings = Settings(
        saves_dir=saves,
        kb_backend="chromadb",
        kb_backend_per_kb={"kb-y": "markdown"},
    ).resolve_paths()
    # Bogus sentinel -> falls through to per-kb mapping.
    assert resolve_backend_name(settings, kb_id="kb-y") == "markdown"


def test_resolve_backend_name_no_kb_id_returns_global(tmp_path: Path) -> None:
    """Legacy callers passing kb_id=None get the global default."""
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        kb_backend="chromadb",
        kb_backend_per_kb={"kb-md": "markdown"},
    ).resolve_paths()
    assert resolve_backend_name(settings) == "chromadb"


# ---------------------------------------------------------------------------
# KnowledgebaseService._backend_for picks the right backend per kb_id
# ---------------------------------------------------------------------------


def test_service_backend_for_uses_per_kb_mapping(tmp_path: Path) -> None:
    """The service caches a markdown backend for kb-md, chromadb for kb-cdb."""
    from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend
    from agent_knowledgebase.backends.markdown_backend import MarkdownWikiBackend
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService

    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        kb_backend="chromadb",
        kb_backend_per_kb={"kb-md": "markdown"},
        embedding_provider="ollama",
    ).resolve_paths()
    svc = KnowledgebaseService(settings)
    kb_md = svc.create_kb(name="kb-md")
    kb_cdb = svc.create_kb(name="kb-cdb")

    # NOTE: per-kb mapping is keyed on kb_id, not kb name. We seeded
    # the env entry with literal "kb-md" — the actual KB id is the
    # uuid that create_kb assigned. To test the routing decision
    # end-to-end against KB ids, re-build settings keyed on the real ids.
    settings_real = Settings(
        saves_dir=saves,
        kb_backend="chromadb",
        kb_backend_per_kb={kb_md.id: "markdown", kb_cdb.id: "chromadb"},
        embedding_provider="ollama",
    ).resolve_paths()
    svc2 = KnowledgebaseService(settings_real)
    # The per-kb cache resolution returns each backend.
    assert isinstance(svc2._backend_for(kb_md.id), MarkdownWikiBackend)
    assert isinstance(svc2._backend_for(kb_cdb.id), ChromadbBackend)
