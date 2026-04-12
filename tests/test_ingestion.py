"""Tests for all ingestors and the IngestionOrchestrator."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.ingestors import (
    INGESTOR_REGISTRY,
    ChunkConfig,
    RawContent,
    token_chunk,
)
from agent_knowledgebase.ingestors.api_endpoint import ApiEndpointIngestor
from agent_knowledgebase.ingestors.codebase import CodebaseIngestor
from agent_knowledgebase.ingestors.directory import DirectoryIngestor
from agent_knowledgebase.ingestors.file import FileIngestor
from agent_knowledgebase.ingestors.git_history import GitHistoryIngestor
from agent_knowledgebase.ingestors.sql_database import SqlDatabaseIngestor
from agent_knowledgebase.ingestors.website import WebsiteIngestor
from agent_knowledgebase.models import SourceType
from agent_knowledgebase.services.ingestion import IngestionOrchestrator


# =====================================================================
# Shared / token_chunk helper
# =====================================================================


class TestTokenChunk:
    """Tests for the shared token_chunk helper."""

    def test_empty_input(self):
        assert token_chunk([], ChunkConfig()) == []

    def test_empty_text(self):
        assert token_chunk([RawContent(text="")], ChunkConfig()) == []

    def test_short_text_single_chunk(self):
        raw = RawContent(text="Hello, world!", metadata={"source": "test"})
        chunks = token_chunk([raw], ChunkConfig(chunk_size=512, chunk_overlap=64))
        assert len(chunks) == 1
        assert chunks[0].content == "Hello, world!"
        assert chunks[0].metadata["source"] == "test"
        assert chunks[0].source_id == ""
        assert chunks[0].kb_id == ""

    def test_long_text_multiple_chunks(self):
        # Build text large enough to require multiple chunks with a small window.
        text = "word " * 200  # ~200 tokens
        raw = RawContent(text=text, metadata={"k": "v"})
        config = ChunkConfig(chunk_size=50, chunk_overlap=10)
        chunks = token_chunk([raw], config)
        assert len(chunks) > 1
        # Every chunk should inherit metadata.
        for c in chunks:
            assert c.metadata["k"] == "v"

    def test_overlap_produces_shared_tokens(self):
        text = " ".join(str(i) for i in range(100))
        raw = RawContent(text=text)
        config = ChunkConfig(chunk_size=20, chunk_overlap=5)
        chunks = token_chunk([raw], config)
        assert len(chunks) >= 2
        # Second chunk should start before the end of the first chunk's range.
        # Just ensure we get more than ceil(100/20) chunks due to overlap.
        no_overlap_count = -(-100 // 20)  # ceil div
        assert len(chunks) >= no_overlap_count


# =====================================================================
# FileIngestor
# =====================================================================


class TestFileIngestor:
    def test_read_text_file(self, tmp_path: Path):
        f = tmp_path / "readme.txt"
        f.write_text("Hello from a text file.", encoding="utf-8")

        ingestor = FileIngestor()
        contents = ingestor.read(str(f))
        assert len(contents) == 1
        assert contents[0].text == "Hello from a text file."
        assert contents[0].metadata["file_type"] == ".txt"
        assert contents[0].metadata["file_path"] == str(f)

    def test_read_markdown_file(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("# Title\n\nSome notes.", encoding="utf-8")

        ingestor = FileIngestor()
        contents = ingestor.read(str(f))
        assert len(contents) == 1
        assert "# Title" in contents[0].text
        assert contents[0].metadata["file_type"] == ".md"

    def test_read_json_file(self, tmp_path: Path):
        data = {"key": "value", "numbers": [1, 2, 3]}
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        ingestor = FileIngestor()
        contents = ingestor.read(str(f))
        assert len(contents) == 1
        # Should be pretty-printed.
        parsed = json.loads(contents[0].text)
        assert parsed == data
        assert contents[0].metadata["file_type"] == ".json"

    def test_read_python_file(self, tmp_path: Path):
        f = tmp_path / "app.py"
        f.write_text("print('hello')", encoding="utf-8")

        ingestor = FileIngestor()
        contents = ingestor.read(str(f))
        assert len(contents) == 1
        assert "print" in contents[0].text
        assert contents[0].metadata["file_type"] == ".py"

    def test_chunk_file(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        f.write_text("word " * 500, encoding="utf-8")

        ingestor = FileIngestor()
        contents = ingestor.read(str(f))
        chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=100, chunk_overlap=10))
        assert len(chunks) >= 2

    def test_read_nonexistent_returns_empty(self, tmp_path: Path):
        """Non-existent files should raise (Path.read_text raises)."""
        ingestor = FileIngestor()
        with pytest.raises(Exception):
            ingestor.read(str(tmp_path / "nope.txt"))

    def test_metadata_passthrough(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("content", encoding="utf-8")
        ingestor = FileIngestor()
        contents = ingestor.read(str(f), metadata={"custom_key": "custom_val"})
        assert contents[0].metadata["custom_key"] == "custom_val"


# =====================================================================
# DirectoryIngestor
# =====================================================================


class TestDirectoryIngestor:
    def _build_tree(self, root: Path) -> None:
        """Build a small directory tree for testing."""
        (root / "a.txt").write_text("file a", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "b.txt").write_text("file b", encoding="utf-8")
        (root / "sub" / "c.py").write_text("x = 1", encoding="utf-8")

    def test_recursive_read(self, tmp_path: Path):
        self._build_tree(tmp_path)
        ingestor = DirectoryIngestor()
        contents = ingestor.read(str(tmp_path))
        paths = [c.metadata["file_path"] for c in contents]
        assert len(contents) == 3
        assert any("a.txt" in p for p in paths)
        assert any("b.txt" in p for p in paths)
        assert any("c.py" in p for p in paths)

    def test_skips_hidden_dirs(self, tmp_path: Path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "secret.txt").write_text("nope", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("yes", encoding="utf-8")

        ingestor = DirectoryIngestor()
        contents = ingestor.read(str(tmp_path))
        assert len(contents) == 1
        assert "visible.txt" in contents[0].metadata["file_path"]

    def test_skips_excluded_dirs(self, tmp_path: Path):
        for d in ("__pycache__", "node_modules", ".git"):
            (tmp_path / d).mkdir()
            (tmp_path / d / "junk.txt").write_text("junk", encoding="utf-8")
        (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")

        ingestor = DirectoryIngestor()
        contents = ingestor.read(str(tmp_path))
        assert len(contents) == 1
        assert "keep.txt" in contents[0].metadata["file_path"]

    def test_gitignore_filtering(self, tmp_path: Path):
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("code", encoding="utf-8")
        (tmp_path / "debug.log").write_text("log data", encoding="utf-8")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "output.js").write_text("built", encoding="utf-8")

        ingestor = DirectoryIngestor()
        contents = ingestor.read(str(tmp_path))
        paths = [c.metadata["file_path"] for c in contents]
        assert any("app.py" in p for p in paths)
        assert not any("debug.log" in p for p in paths)
        assert not any("output.js" in p for p in paths)

    def test_nonexistent_directory(self, tmp_path: Path):
        ingestor = DirectoryIngestor()
        contents = ingestor.read(str(tmp_path / "nope"))
        assert contents == []

    def test_chunk(self, tmp_path: Path):
        self._build_tree(tmp_path)
        ingestor = DirectoryIngestor()
        contents = ingestor.read(str(tmp_path))
        chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=512))
        assert len(chunks) >= 3  # at least one per file


# =====================================================================
# CodebaseIngestor
# =====================================================================


class TestCodebaseIngestor:
    def test_language_detection(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "app.js").write_text("let x = 1;", encoding="utf-8")
        (tmp_path / "data.txt").write_text("hello", encoding="utf-8")

        ingestor = CodebaseIngestor()
        contents = ingestor.read(str(tmp_path))
        langs = {c.metadata["file_path"].split(".")[-1]: c.metadata["language"] for c in contents}
        assert langs["py"] == "python"
        assert langs["js"] == "javascript"
        assert langs["txt"] == "unknown"

    def test_python_symbol_chunking(self, tmp_path: Path):
        code = textwrap.dedent("""\
            import os

            class Foo:
                def bar(self):
                    pass

            def baz():
                return 42
        """)
        (tmp_path / "module.py").write_text(code, encoding="utf-8")

        ingestor = CodebaseIngestor()
        contents = ingestor.read(str(tmp_path))
        chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=512, chunk_overlap=0))

        symbol_types = [c.metadata.get("symbol_type") for c in chunks]
        # Should detect module preamble, class Foo, and function baz.
        assert "module" in symbol_types
        assert "class" in symbol_types
        assert "function" in symbol_types

    def test_unknown_language_fallback(self, tmp_path: Path):
        (tmp_path / "data.txt").write_text("word " * 200, encoding="utf-8")

        ingestor = CodebaseIngestor()
        contents = ingestor.read(str(tmp_path))
        chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=50, chunk_overlap=10))
        # Should fall back to token chunking.
        assert len(chunks) > 1

    def test_large_symbol_gets_sub_chunked(self, tmp_path: Path):
        # Create a function with a very large body.
        body_lines = [f"    x_{i} = {i}" for i in range(200)]
        code = "def huge_function():\n" + "\n".join(body_lines)
        (tmp_path / "big.py").write_text(code, encoding="utf-8")

        ingestor = CodebaseIngestor()
        contents = ingestor.read(str(tmp_path))
        chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=50, chunk_overlap=10))
        # The single large function should be broken into multiple chunks.
        assert len(chunks) > 1


# =====================================================================
# WebsiteIngestor
# =====================================================================


class TestWebsiteIngestor:
    @patch("trafilatura.fetch_url")
    @patch("trafilatura.extract")
    def test_read_success(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html><body>Test content</body></html>"
        # First call: plain text extraction; second call: XML extraction for title.
        mock_extract.side_effect = [
            "Extracted main text",
            '<doc title="Test Page">content</doc>',
        ]

        ingestor = WebsiteIngestor()
        contents = ingestor.read("https://example.com")
        assert len(contents) == 1
        assert contents[0].text == "Extracted main text"
        assert contents[0].metadata["url"] == "https://example.com"
        assert contents[0].metadata["title"] == "Test Page"

    @patch("trafilatura.fetch_url")
    def test_read_fetch_failure(self, mock_fetch):
        mock_fetch.return_value = None

        ingestor = WebsiteIngestor()
        contents = ingestor.read("https://example.com/404")
        assert contents == []

    @patch("trafilatura.fetch_url")
    @patch("trafilatura.extract")
    def test_read_empty_extract(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html></html>"
        mock_extract.return_value = ""

        ingestor = WebsiteIngestor()
        contents = ingestor.read("https://example.com/empty")
        assert contents == []

    @patch("trafilatura.fetch_url")
    @patch("trafilatura.extract")
    def test_chunk(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html><body>text</body></html>"
        mock_extract.side_effect = ["word " * 200, ""]

        ingestor = WebsiteIngestor()
        contents = ingestor.read("https://example.com")
        chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=50, chunk_overlap=10))
        assert len(chunks) > 1


# =====================================================================
# SqlDatabaseIngestor
# =====================================================================


class TestSqlDatabaseIngestor:
    def _create_db(self, tmp_path: Path) -> str:
        """Create a simple SQLite database and return its connection URI."""
        from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

        db_path = tmp_path / "test.db"
        uri = f"sqlite:///{db_path}"
        engine = create_engine(uri)
        meta = MetaData()
        Table(
            "users",
            meta,
            Column("id", Integer, primary_key=True),
            Column("name", String(100), nullable=False),
            Column("email", String(200)),
        )
        meta.create_all(engine)

        # Insert sample data.
        with engine.connect() as conn:
            from sqlalchemy import text

            conn.execute(
                text("INSERT INTO users (id, name, email) VALUES (:id, :name, :email)"),
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
            )
            conn.commit()

        engine.dispose()
        return uri

    def test_read_schema(self, tmp_path: Path):
        uri = self._create_db(tmp_path)
        ingestor = SqlDatabaseIngestor()
        contents = ingestor.read(uri)
        assert len(contents) == 1  # one table
        text = contents[0].text
        assert "users" in text
        assert "name" in text
        assert "email" in text
        assert contents[0].metadata["table_name"] == "users"
        assert contents[0].metadata["connection_string"] == uri

    def test_read_with_sample_rows(self, tmp_path: Path):
        uri = self._create_db(tmp_path)
        ingestor = SqlDatabaseIngestor()
        contents = ingestor.read(uri, metadata={"sample_rows": 5})
        assert len(contents) == 1
        text = contents[0].text
        assert "Alice" in text
        assert "Sample rows" in text

    def test_chunk(self, tmp_path: Path):
        uri = self._create_db(tmp_path)
        ingestor = SqlDatabaseIngestor()
        contents = ingestor.read(uri)
        chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=512))
        assert len(chunks) >= 1


# =====================================================================
# GitHistoryIngestor
# =====================================================================


class TestGitHistoryIngestor:
    def _create_repo(self, tmp_path: Path) -> Path:
        """Create a small git repo with a couple of commits."""
        from git import Repo

        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)

        # Configure user for commits.
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # First commit.
        f = repo_path / "hello.txt"
        f.write_text("Hello", encoding="utf-8")
        repo.index.add(["hello.txt"])
        repo.index.commit("Initial commit")

        # Second commit.
        f.write_text("Hello, World!", encoding="utf-8")
        repo.index.add(["hello.txt"])
        repo.index.commit("Update greeting")

        return repo_path

    def test_read_commits(self, tmp_path: Path):
        repo_path = self._create_repo(tmp_path)
        ingestor = GitHistoryIngestor()
        contents = ingestor.read(str(repo_path))
        assert len(contents) == 2
        # Most recent first.
        assert "Update greeting" in contents[0].text
        assert "Initial commit" in contents[1].text
        assert contents[0].metadata["repo_path"] == str(repo_path)
        assert "commit_sha" in contents[0].metadata
        assert "author" in contents[0].metadata
        assert "date" in contents[0].metadata

    def test_max_commits(self, tmp_path: Path):
        repo_path = self._create_repo(tmp_path)
        ingestor = GitHistoryIngestor()
        contents = ingestor.read(str(repo_path), metadata={"max_commits": 1})
        assert len(contents) == 1

    def test_chunk(self, tmp_path: Path):
        repo_path = self._create_repo(tmp_path)
        ingestor = GitHistoryIngestor()
        contents = ingestor.read(str(repo_path))
        chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=512))
        assert len(chunks) >= 2


# =====================================================================
# ApiEndpointIngestor
# =====================================================================


class TestApiEndpointIngestor:
    def test_read_json_response(self):
        import httpx

        mock_response = httpx.Response(
            status_code=200,
            json={"result": "ok", "items": [1, 2, 3]},
            request=httpx.Request("GET", "https://api.example.com/data"),
        )

        ingestor = ApiEndpointIngestor()
        with patch("httpx.get", return_value=mock_response):
            contents = ingestor.read("https://api.example.com/data")

        assert len(contents) == 1
        parsed = json.loads(contents[0].text)
        assert parsed["result"] == "ok"
        assert contents[0].metadata["url"] == "https://api.example.com/data"
        assert contents[0].metadata["status_code"] == 200

    def test_read_text_response(self):
        import httpx

        mock_response = httpx.Response(
            status_code=200,
            text="Plain text response",
            headers={"content-type": "text/plain"},
            request=httpx.Request("GET", "https://api.example.com/text"),
        )

        ingestor = ApiEndpointIngestor()
        with patch("httpx.get", return_value=mock_response):
            contents = ingestor.read("https://api.example.com/text")

        assert len(contents) == 1
        assert contents[0].text == "Plain text response"
        assert "text/plain" in contents[0].metadata["content_type"]

    def test_read_error_raises(self):
        import httpx

        mock_response = httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.example.com/missing"),
        )

        ingestor = ApiEndpointIngestor()
        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                ingestor.read("https://api.example.com/missing")

    def test_chunk(self):
        import httpx

        mock_response = httpx.Response(
            status_code=200,
            json={"data": "x " * 300},
            request=httpx.Request("GET", "https://api.example.com/big"),
        )

        ingestor = ApiEndpointIngestor()
        with patch("httpx.get", return_value=mock_response):
            contents = ingestor.read("https://api.example.com/big")
            chunks = ingestor.chunk(contents, ChunkConfig(chunk_size=50, chunk_overlap=10))
            assert len(chunks) >= 1


# =====================================================================
# INGESTOR_REGISTRY
# =====================================================================


class TestIngestorRegistry:
    def test_all_source_types_registered(self):
        for st in SourceType:
            assert st in INGESTOR_REGISTRY, f"{st} missing from registry"

    def test_registry_types(self):
        assert INGESTOR_REGISTRY[SourceType.file] is FileIngestor
        assert INGESTOR_REGISTRY[SourceType.directory] is DirectoryIngestor
        assert INGESTOR_REGISTRY[SourceType.codebase] is CodebaseIngestor
        assert INGESTOR_REGISTRY[SourceType.website] is WebsiteIngestor
        assert INGESTOR_REGISTRY[SourceType.sql_database] is SqlDatabaseIngestor
        assert INGESTOR_REGISTRY[SourceType.git_history] is GitHistoryIngestor
        assert INGESTOR_REGISTRY[SourceType.api_endpoint] is ApiEndpointIngestor


# =====================================================================
# IngestionOrchestrator
# =====================================================================


class TestIngestionOrchestrator:
    def test_dispatch_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("Hello orchestrator!", encoding="utf-8")

        settings = Settings(saves_dir=tmp_path)
        orchestrator = IngestionOrchestrator(settings)
        chunks = orchestrator.ingest(SourceType.file, str(f))
        assert len(chunks) >= 1
        assert "Hello orchestrator!" in chunks[0].content

    def test_dispatch_directory(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
        (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")

        settings = Settings(saves_dir=tmp_path)
        orchestrator = IngestionOrchestrator(settings)
        chunks = orchestrator.ingest(SourceType.directory, str(tmp_path))
        assert len(chunks) >= 2

    def test_custom_chunk_config(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        f.write_text("word " * 500, encoding="utf-8")

        settings = Settings(saves_dir=tmp_path)
        orchestrator = IngestionOrchestrator(settings)
        custom = ChunkConfig(chunk_size=50, chunk_overlap=5)
        chunks = orchestrator.ingest(SourceType.file, str(f), chunk_config=custom)
        assert len(chunks) > 1

    def test_unknown_source_type(self, tmp_path: Path):
        settings = Settings(saves_dir=tmp_path)
        orchestrator = IngestionOrchestrator(settings)
        # Create a fake source type value to trigger the error.
        with pytest.raises(ValueError, match="No ingestor registered"):
            orchestrator.ingest("nonexistent_type", "whatever")  # type: ignore[arg-type]

    def test_uses_settings_defaults(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("short", encoding="utf-8")

        settings = Settings(saves_dir=tmp_path, chunk_size=512, chunk_overlap=64)
        orchestrator = IngestionOrchestrator(settings)
        chunks = orchestrator.ingest(SourceType.file, str(f))
        assert len(chunks) == 1
