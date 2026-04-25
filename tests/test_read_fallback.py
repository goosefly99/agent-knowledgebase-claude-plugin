"""Phase 4 — markdown -> chromadb read-fallback semantics.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[4].tasks[2] — "Read-fallback: if
        Settings.kb_backend='markdown' but <kb-name>/wiki/ does not
        exist while <kb-name>/vector_store/ does, ChromadbBackend
        serves the query — silent fallback, structured stderr line
        emitted."

Test surface
------------

* When wiki/ is missing and chroma/ exists, MarkdownWikiBackend.search
  delegates to ChromadbBackend (proven via mocking the chromadb
  delegate).
* The structured stderr_log line carries all 11 required fields plus
  ``error_code='MARKDOWN_READ_FALLBACK_TO_CHROMADB'`` and
  ``phase='read_fallback'``.
* When wiki/ exists, no fallback fires — markdown serves natively.
* When neither wiki/ nor chroma/ exist, no fallback (markdown returns
  empty FTS results).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_knowledgebase.backends.markdown_backend import MarkdownWikiBackend
from agent_knowledgebase.config import Settings


@pytest.fixture()
def markdown_backend(tmp_path: Path) -> MarkdownWikiBackend:
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(saves_dir=saves, kb_backend="markdown").resolve_paths()
    return MarkdownWikiBackend(settings, service=None)


# ---------------------------------------------------------------------------
# Fallback fires when wiki/ missing AND chroma/ exists
# ---------------------------------------------------------------------------


def test_search_falls_back_to_chromadb_when_wiki_missing_and_chroma_present(
    markdown_backend: MarkdownWikiBackend,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Markdown.search routes through ChromadbBackend when wiki/ is missing."""
    kb_id = "kb-fallback"
    # Simulate a v0.6.0 KB that has a chroma/ collection but no wiki/.
    kb_root = markdown_backend._kb_root(kb_id)
    (kb_root / "chroma").mkdir(parents=True, exist_ok=True)

    # Patch ChromadbBackend.search so we observe delegation without
    # spinning up real chromadb.
    sentinel = [{
        "content": "fallback-result",
        "source_id": "src-1",
        "source_type": "file",
        "score": 0.9,
        "metadata": {"backend": "chromadb"},
    }]
    with patch(
        "agent_knowledgebase.backends.chromadb_backend.ChromadbBackend.search",
        return_value=sentinel,
    ) as mock_search:
        out = markdown_backend.search(kb_id=kb_id, text="anything", top_k=5)
    assert mock_search.called, "ChromadbBackend.search must serve the fallback"
    assert out == sentinel

    # And a structured stderr_log line landed with the right fields.
    err = capsys.readouterr().err
    found = False
    for line in err.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("error_code") == "MARKDOWN_READ_FALLBACK_TO_CHROMADB":
            found = True
            # 11-field schema must be intact.
            for key in (
                "kb_id",
                "op",
                "phase",
                "elapsed_ms",
                "rows_in",
                "rows_ok",
                "rows_skipped",
                "rows_failed",
                "dedup_policy",
                "request_id",
                "tool_caller_version",
            ):
                assert key in payload, f"missing field {key}"
            assert payload["phase"] == "read_fallback"
            assert payload["op"] == "kb_query"
            assert payload["kb_id"] == kb_id
            break
    assert found, "expected MARKDOWN_READ_FALLBACK_TO_CHROMADB stderr line"


def test_query_falls_back_to_chromadb_when_wiki_missing_and_chroma_present(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """Same fallback semantics on the vector path."""
    kb_id = "kb-q-fallback"
    kb_root = markdown_backend._kb_root(kb_id)
    (kb_root / "chroma").mkdir(parents=True, exist_ok=True)

    sentinel = [{"content": "x", "source_id": "y", "source_type": "z", "score": 1.0, "metadata": {}}]
    with patch(
        "agent_knowledgebase.backends.chromadb_backend.ChromadbBackend.query",
        return_value=sentinel,
    ) as mock_query:
        out = markdown_backend.query(kb_id=kb_id, text="x", top_k=5)
    assert mock_query.called
    assert out == sentinel


# ---------------------------------------------------------------------------
# Fallback ALSO accepts the spec's vector_store/ alias
# ---------------------------------------------------------------------------


def test_fallback_also_recognizes_vector_store_alias(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """The spec uses ``vector_store/`` as the alias for ``chroma/``."""
    kb_id = "kb-vs-alias"
    kb_root = markdown_backend._kb_root(kb_id)
    (kb_root / "vector_store").mkdir(parents=True, exist_ok=True)

    sentinel = [{"content": "vs-hit", "source_id": "y", "source_type": "z", "score": 1.0, "metadata": {}}]
    with patch(
        "agent_knowledgebase.backends.chromadb_backend.ChromadbBackend.search",
        return_value=sentinel,
    ) as mock_search:
        out = markdown_backend.search(kb_id=kb_id, text="x", top_k=5)
    assert mock_search.called
    assert out == sentinel


# ---------------------------------------------------------------------------
# No fallback when wiki/ has pages
# ---------------------------------------------------------------------------


def test_no_fallback_when_wiki_layout_present(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """If wiki/ exists with at least one page, markdown serves the query."""
    kb_id = "kb-no-fallback"
    # Seed the wiki layout the natural way (via index()).
    markdown_backend.index(
        kb_id=kb_id,
        documents=[{
            "id": "src-x",
            "content": "alpha beta gamma",
            "metadata": {
                "source_type": "file",
                "title": "x",
                "uri": "/tmp/x.txt",
                "dedup_key": None,
            },
        }],
    )
    # Even if a chroma/ dir exists alongside, the wiki/ layout takes
    # precedence (markdown serves the query natively).
    kb_root = markdown_backend._kb_root(kb_id)
    (kb_root / "chroma").mkdir(parents=True, exist_ok=True)

    with patch(
        "agent_knowledgebase.backends.chromadb_backend.ChromadbBackend.search",
        side_effect=AssertionError("fallback should NOT fire"),
    ):
        out = markdown_backend.search(kb_id=kb_id, text="alpha", top_k=5)
    assert out, "expected at least one FTS5 hit from native markdown"


def test_no_fallback_when_neither_layout_present(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """If neither wiki/ nor chroma/ exists, markdown returns [] (no fallback)."""
    kb_id = "kb-empty"
    with patch(
        "agent_knowledgebase.backends.chromadb_backend.ChromadbBackend.search",
        side_effect=AssertionError("fallback should NOT fire on truly-empty KB"),
    ):
        out = markdown_backend.search(kb_id=kb_id, text="anything", top_k=5)
    assert out == []
