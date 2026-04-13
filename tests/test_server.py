"""Tests for MCP server tool registration, input validation, and response format."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_knowledgebase.models import (
    Knowledgebase,
    PageType,
    PipelinePhase,
    PipelineRun,
    RunStatus,
    Source,
    SourceStatus,
    SourceType,
    WikiPage,
)
from agent_knowledgebase.server import (
    _serialize_dataclass,
    _serialize_dataclass_list,
    _serialize_model_list,
    kb_create,
    kb_delete,
    kb_export,
    kb_get_links,
    kb_get_page,
    kb_get_source,
    kb_info,
    kb_ingest,
    kb_ingest_batch,
    kb_lint,
    kb_lint_fix,
    kb_list,
    kb_list_pages,
    kb_list_sources,
    kb_pipeline_status,
    kb_query,
    kb_rebuild_index,
    kb_remove_source,
    kb_search,
    kb_update_source,
    mcp,
)
from agent_knowledgebase.services.lint import LintIssue, LintReport
from agent_knowledgebase.services.query import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_kb(**overrides) -> Knowledgebase:
    defaults = dict(
        id="kb-1", name="Test KB", description="A test KB",
        created_at=_NOW, updated_at=_NOW,
        source_count=2, page_count=5,
    )
    defaults.update(overrides)
    return Knowledgebase(**defaults)


def _make_source(**overrides) -> Source:
    defaults = dict(
        id="src-1", kb_id="kb-1", source_type=SourceType.file,
        uri="/tmp/test.txt", metadata={}, status=SourceStatus.ingested,
        chunk_count=10, ingested_at=_NOW,
    )
    defaults.update(overrides)
    return Source(**defaults)


def _make_page(**overrides) -> WikiPage:
    defaults = dict(
        id="page-1", kb_id="kb-1", title="Test Page",
        content="# Test\n\nContent here.", page_type=PageType.entity,
        source_ids=["src-1"], tags=["test"],
        created_at=_NOW, updated_at=_NOW,
    )
    defaults.update(overrides)
    return WikiPage(**defaults)


def _make_run(**overrides) -> PipelineRun:
    defaults = dict(
        id="run-1", kb_id="kb-1", source_id="src-1",
        phase=PipelinePhase.finalize, status=RunStatus.completed,
        started_at=_NOW, completed_at=_NOW,
    )
    defaults.update(overrides)
    return PipelineRun(**defaults)


def _make_search_result(**overrides) -> SearchResult:
    defaults = dict(
        content="matched text", source_id="chunk-1",
        source_type="chunk", score=0.95,
        metadata={"key": "value"},
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


def _make_lint_report(**overrides) -> LintReport:
    defaults = dict(
        issues=[], checked_at=_NOW, page_count=3, link_count=5,
    )
    defaults.update(overrides)
    return LintReport(**defaults)


def _make_lint_issue(**overrides) -> LintIssue:
    defaults = dict(
        type="orphan_page", severity="warning",
        page_id="page-1", message="Page has no inbound links.",
        auto_fixable=False,
    )
    defaults.update(overrides)
    return LintIssue(**defaults)


@pytest.fixture()
def mock_service():
    """Return a MagicMock standing in for KnowledgebaseService."""
    return MagicMock()


@pytest.fixture(autouse=True)
def _patch_get_service(mock_service):
    """Patch _get_service so every tool function uses the mock."""
    with patch("agent_knowledgebase.server._get_service", return_value=mock_service):
        yield


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Verify all 20 tools are registered on the FastMCP instance."""

    EXPECTED_TOOLS = {
        "kb_create",
        "kb_delete",
        "kb_ingest",
        "kb_ingest_batch",
        "kb_update_source",
        "kb_remove_source",
        "kb_lint",
        "kb_lint_fix",
        "kb_rebuild_index",
        "kb_export",
        "kb_list",
        "kb_info",
        "kb_query",
        "kb_search",
        "kb_get_page",
        "kb_list_pages",
        "kb_get_source",
        "kb_list_sources",
        "kb_get_links",
        "kb_pipeline_status",
    }

    def test_all_tools_registered(self):
        """All 20 tools must be registered on the mcp instance."""
        registered = set(mcp._tool_manager._tools.keys())
        assert self.EXPECTED_TOOLS.issubset(registered), (
            f"Missing tools: {self.EXPECTED_TOOLS - registered}"
        )

    def test_tool_count(self):
        """Exactly 20 tools should be registered."""
        registered = set(mcp._tool_manager._tools.keys())
        tool_overlap = registered & self.EXPECTED_TOOLS
        assert len(tool_overlap) == 20


# ---------------------------------------------------------------------------
# Write-Path Tools
# ---------------------------------------------------------------------------


class TestKbCreate:
    def test_returns_json(self, mock_service):
        mock_service.create_kb.return_value = _make_kb()
        result = kb_create("Test KB", "A test KB")
        parsed = json.loads(result)
        assert parsed["name"] == "Test KB"
        assert parsed["id"] == "kb-1"
        mock_service.create_kb.assert_called_once_with("Test KB", "A test KB")

    def test_default_description(self, mock_service):
        mock_service.create_kb.return_value = _make_kb(description="")
        kb_create("Test KB")
        mock_service.create_kb.assert_called_once_with("Test KB", "")


class TestKbDelete:
    def test_returns_confirmation(self, mock_service):
        result = kb_delete("kb-1")
        parsed = json.loads(result)
        assert parsed["status"] == "deleted"
        assert parsed["kb_id"] == "kb-1"
        mock_service.delete_kb.assert_called_once_with("kb-1")


class TestKbIngest:
    def test_valid_source_type(self, mock_service):
        mock_service.ingest_source.return_value = _make_source()
        result = kb_ingest("kb-1", "file", "/tmp/test.txt")
        parsed = json.loads(result)
        assert parsed["source_type"] == "file"
        mock_service.ingest_source.assert_called_once_with(
            "kb-1", SourceType.file, "/tmp/test.txt", {}
        )

    def test_with_metadata(self, mock_service):
        mock_service.ingest_source.return_value = _make_source()
        meta = json.dumps({"lang": "python"})
        kb_ingest("kb-1", "file", "/tmp/test.py", meta)
        mock_service.ingest_source.assert_called_once_with(
            "kb-1", SourceType.file, "/tmp/test.py", {"lang": "python"}
        )

    def test_invalid_source_type(self, mock_service):
        with pytest.raises(ValueError, match="'invalid'"):
            kb_ingest("kb-1", "invalid", "/tmp/test.txt")

    def test_invalid_metadata_json(self, mock_service):
        with pytest.raises(json.JSONDecodeError):
            kb_ingest("kb-1", "file", "/tmp/test.txt", "not-json")


class TestKbIngestBatch:
    def test_multiple_sources(self, mock_service):
        src1 = _make_source(id="src-1")
        src2 = _make_source(id="src-2", uri="/tmp/other.txt")
        mock_service.ingest_source.side_effect = [src1, src2]
        sources_json = json.dumps([
            {"source_type": "file", "uri": "/tmp/test.txt"},
            {"source_type": "file", "uri": "/tmp/other.txt"},
        ])
        result = kb_ingest_batch("kb-1", sources_json)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert mock_service.ingest_source.call_count == 2

    def test_empty_batch(self, mock_service):
        result = kb_ingest_batch("kb-1", "[]")
        parsed = json.loads(result)
        assert parsed == []

    def test_invalid_sources_json(self, mock_service):
        with pytest.raises(json.JSONDecodeError):
            kb_ingest_batch("kb-1", "not-json")


class TestKbUpdateSource:
    def test_returns_updated_source(self, mock_service):
        mock_service.update_source.return_value = _make_source()
        result = kb_update_source("src-1")
        parsed = json.loads(result)
        assert parsed["id"] == "src-1"
        mock_service.update_source.assert_called_once_with("src-1")


class TestKbRemoveSource:
    def test_returns_confirmation(self, mock_service):
        result = kb_remove_source("src-1")
        parsed = json.loads(result)
        assert parsed["status"] == "removed"
        assert parsed["source_id"] == "src-1"
        mock_service.remove_source.assert_called_once_with("src-1")


class TestKbLint:
    def test_returns_lint_report(self, mock_service):
        issue = _make_lint_issue()
        report = _make_lint_report(issues=[issue], page_count=3, link_count=5)
        mock_service.lint.return_value = report
        result = kb_lint("kb-1")
        parsed = json.loads(result)
        assert parsed["page_count"] == 3
        assert parsed["link_count"] == 5
        assert len(parsed["issues"]) == 1
        assert parsed["issues"][0]["type"] == "orphan_page"


class TestKbLintFix:
    def test_returns_fixed_issues(self, mock_service):
        issue = _make_lint_issue(auto_fixable=True, type="index_drift")
        mock_service.lint_fix.return_value = [issue]
        result = kb_lint_fix("kb-1")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["type"] == "index_drift"
        assert parsed[0]["auto_fixable"] is True

    def test_no_fixable_issues(self, mock_service):
        mock_service.lint_fix.return_value = []
        result = kb_lint_fix("kb-1")
        parsed = json.loads(result)
        assert parsed == []


class TestKbRebuildIndex:
    def test_returns_index_page(self, mock_service):
        index_page = _make_page(title="Index", page_type=PageType.index)
        mock_service.rebuild_index.return_value = index_page
        result = kb_rebuild_index("kb-1")
        parsed = json.loads(result)
        assert parsed["title"] == "Index"
        assert parsed["page_type"] == "index"


class TestKbExport:
    def test_returns_file_paths(self, mock_service):
        mock_service.export.return_value = [
            Path("/tmp/export/page1.md"),
            Path("/tmp/export/page2.md"),
        ]
        result = kb_export("kb-1", "/tmp/export")
        parsed = json.loads(result)
        assert len(parsed) == 2
        # Path separators may vary by OS; just check the filenames are present
        assert any("page1.md" in p for p in parsed)
        assert any("page2.md" in p for p in parsed)
        mock_service.export.assert_called_once_with("kb-1", Path("/tmp/export"))

    def test_empty_export(self, mock_service):
        mock_service.export.return_value = []
        result = kb_export("kb-1", "/tmp/export")
        assert json.loads(result) == []


# ---------------------------------------------------------------------------
# Read-Path Tools
# ---------------------------------------------------------------------------


class TestKbList:
    def test_returns_array(self, mock_service):
        mock_service.list_kbs.return_value = [_make_kb(), _make_kb(id="kb-2", name="Second")]
        result = kb_list()
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "kb-1"
        assert parsed[1]["id"] == "kb-2"

    def test_empty_list(self, mock_service):
        mock_service.list_kbs.return_value = []
        result = kb_list()
        assert json.loads(result) == []


class TestKbInfo:
    def test_returns_kb_details(self, mock_service):
        mock_service.get_kb.return_value = _make_kb(source_count=3, page_count=7)
        result = kb_info("kb-1")
        parsed = json.loads(result)
        assert parsed["source_count"] == 3
        assert parsed["page_count"] == 7

    def test_not_found_raises(self, mock_service):
        mock_service.get_kb.return_value = None
        with pytest.raises(ValueError, match="not found"):
            kb_info("nonexistent")


class TestKbQuery:
    def test_returns_search_results(self, mock_service):
        mock_service.query.return_value = [_make_search_result()]
        result = kb_query("kb-1", "test query", 5)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["score"] == 0.95
        assert parsed[0]["content"] == "matched text"
        mock_service.query.assert_called_once_with("kb-1", "test query", 5)

    def test_default_top_k(self, mock_service):
        mock_service.query.return_value = []
        kb_query("kb-1", "test")
        mock_service.query.assert_called_once_with("kb-1", "test", 10)


class TestKbSearch:
    def test_returns_search_results(self, mock_service):
        mock_service.search.return_value = [_make_search_result(score=0.8)]
        result = kb_search("kb-1", "keyword", 3)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["score"] == 0.8
        mock_service.search.assert_called_once_with("kb-1", "keyword", 3)


class TestKbGetPage:
    def test_returns_page(self, mock_service):
        mock_service.get_page.return_value = _make_page()
        result = kb_get_page("page-1")
        parsed = json.loads(result)
        assert parsed["title"] == "Test Page"
        assert parsed["page_type"] == "entity"

    def test_not_found_raises(self, mock_service):
        mock_service.get_page.return_value = None
        with pytest.raises(ValueError, match="not found"):
            kb_get_page("nonexistent")


class TestKbListPages:
    def test_all_pages(self, mock_service):
        mock_service.list_pages.return_value = [_make_page(), _make_page(id="page-2")]
        result = kb_list_pages("kb-1")
        parsed = json.loads(result)
        assert len(parsed) == 2
        mock_service.list_pages.assert_called_once_with("kb-1", None)

    def test_filtered_by_type(self, mock_service):
        mock_service.list_pages.return_value = [_make_page(page_type=PageType.summary)]
        result = kb_list_pages("kb-1", "summary")
        parsed = json.loads(result)
        assert len(parsed) == 1
        mock_service.list_pages.assert_called_once_with("kb-1", PageType.summary)

    def test_invalid_page_type(self, mock_service):
        with pytest.raises(ValueError, match="'invalid'"):
            kb_list_pages("kb-1", "invalid")


class TestKbGetSource:
    def test_returns_source(self, mock_service):
        mock_service.get_source.return_value = _make_source()
        result = kb_get_source("src-1")
        parsed = json.loads(result)
        assert parsed["id"] == "src-1"
        assert parsed["source_type"] == "file"

    def test_not_found_raises(self, mock_service):
        mock_service.get_source.return_value = None
        with pytest.raises(ValueError, match="not found"):
            kb_get_source("nonexistent")


class TestKbListSources:
    def test_returns_array(self, mock_service):
        mock_service.list_sources.return_value = [_make_source()]
        result = kb_list_sources("kb-1")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["uri"] == "/tmp/test.txt"

    def test_empty_list(self, mock_service):
        mock_service.list_sources.return_value = []
        result = kb_list_sources("kb-1")
        assert json.loads(result) == []


class TestKbGetLinks:
    def test_outbound_links(self, mock_service):
        linked = _make_page(id="page-2", title="Linked Page")
        mock_service.get_links.return_value = [linked]
        result = kb_get_links("page-1", "outbound")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "page-2"
        mock_service.get_links.assert_called_once_with("page-1", "outbound")

    def test_inbound_links(self, mock_service):
        mock_service.get_links.return_value = []
        result = kb_get_links("page-1", "inbound")
        parsed = json.loads(result)
        assert parsed == []
        mock_service.get_links.assert_called_once_with("page-1", "inbound")

    def test_default_direction(self, mock_service):
        mock_service.get_links.return_value = []
        kb_get_links("page-1")
        mock_service.get_links.assert_called_once_with("page-1", "outbound")

    def test_invalid_direction(self, mock_service):
        with pytest.raises(ValueError, match="Invalid direction"):
            kb_get_links("page-1", "sideways")


class TestKbPipelineStatus:
    def test_returns_runs(self, mock_service):
        mock_service.get_pipeline_status.return_value = [_make_run()]
        result = kb_pipeline_status("kb-1")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "run-1"
        assert parsed[0]["status"] == "completed"

    def test_empty_status(self, mock_service):
        mock_service.get_pipeline_status.return_value = []
        result = kb_pipeline_status("kb-1")
        assert json.loads(result) == []


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------


class TestSerializationHelpers:
    def test_serialize_model_list_empty(self):
        assert _serialize_model_list([]) == "[]"

    def test_serialize_model_list_single(self):
        kb = _make_kb()
        result = _serialize_model_list([kb])
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "kb-1"

    def test_serialize_model_list_multiple(self):
        kbs = [_make_kb(id="kb-1"), _make_kb(id="kb-2")]
        result = _serialize_model_list(kbs)
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_serialize_dataclass(self):
        sr = _make_search_result()
        result = _serialize_dataclass(sr)
        parsed = json.loads(result)
        assert parsed["score"] == 0.95
        assert parsed["content"] == "matched text"

    def test_serialize_dataclass_list(self):
        items = [_make_search_result(), _make_search_result(score=0.5)]
        result = _serialize_dataclass_list(items)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[1]["score"] == 0.5

    def test_serialize_dataclass_list_empty(self):
        result = _serialize_dataclass_list([])
        assert json.loads(result) == []

    def test_serialize_lint_report_with_datetime(self):
        """Datetime fields in dataclasses serialize via default=str."""
        report = _make_lint_report()
        result = _serialize_dataclass(report)
        parsed = json.loads(result)
        assert "2025" in parsed["checked_at"]


# ---------------------------------------------------------------------------
# Response Format Consistency
# ---------------------------------------------------------------------------


class TestResponseFormat:
    """All tool return values must be valid JSON strings."""

    def test_all_tools_return_str(self, mock_service):
        """Smoke-test that each tool function returns a string."""
        # Set up return values for all service methods
        mock_service.create_kb.return_value = _make_kb()
        mock_service.list_kbs.return_value = []
        mock_service.get_kb.return_value = _make_kb()
        mock_service.ingest_source.return_value = _make_source()
        mock_service.update_source.return_value = _make_source()
        mock_service.get_page.return_value = _make_page()
        mock_service.list_pages.return_value = []
        mock_service.get_source.return_value = _make_source()
        mock_service.list_sources.return_value = []
        mock_service.get_links.return_value = []
        mock_service.query.return_value = []
        mock_service.search.return_value = []
        mock_service.lint.return_value = _make_lint_report()
        mock_service.lint_fix.return_value = []
        mock_service.rebuild_index.return_value = _make_page(page_type=PageType.index)
        mock_service.export.return_value = []
        mock_service.get_pipeline_status.return_value = []

        # Call every tool and verify it returns a str that is valid JSON
        calls = [
            lambda: kb_create("test"),
            lambda: kb_delete("kb-1"),
            lambda: kb_ingest("kb-1", "file", "/tmp/f"),
            lambda: kb_ingest_batch("kb-1", "[]"),
            lambda: kb_update_source("src-1"),
            lambda: kb_remove_source("src-1"),
            lambda: kb_lint("kb-1"),
            lambda: kb_lint_fix("kb-1"),
            lambda: kb_rebuild_index("kb-1"),
            lambda: kb_export("kb-1", "/tmp/out"),
            lambda: kb_list(),
            lambda: kb_info("kb-1"),
            lambda: kb_query("kb-1", "test"),
            lambda: kb_search("kb-1", "test"),
            lambda: kb_get_page("page-1"),
            lambda: kb_list_pages("kb-1"),
            lambda: kb_get_source("src-1"),
            lambda: kb_list_sources("kb-1"),
            lambda: kb_get_links("page-1"),
            lambda: kb_pipeline_status("kb-1"),
        ]

        for i, call in enumerate(calls):
            result = call()
            assert isinstance(result, str), f"Tool #{i} did not return str"
            json.loads(result)  # must be valid JSON


class TestKbConfigPath:
    def test_returns_both_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import json as _json
        from agent_knowledgebase import server as srv
        saves = tmp_path / "saves"
        saves.mkdir()
        user_cfg = tmp_path / "user.json"
        user_cfg.write_text("{}")
        monkeypatch.setenv("AGENT_KB_SAVES_DIR", str(saves))
        monkeypatch.setenv("AGENT_KB_USER_CONFIG", str(user_cfg))
        monkeypatch.setenv("AGENT_KB_PROJECT_CONFIG", str(tmp_path / "proj.json"))
        result = _json.loads(srv.kb_config_path())
        assert result["user"]["path"] == str(user_cfg)
        assert result["user"]["exists"] is True
        assert result["user"]["resolved_via"] == "env_override"
        assert result["project"]["exists"] is False
        assert result["project"]["resolved_via"] == "env_override"
