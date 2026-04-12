"""DirectoryIngestor -- recursively walk a directory, respecting .gitignore."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from agent_knowledgebase.ingestors import ChunkConfig, Chunk, RawContent, token_chunk
from agent_knowledgebase.ingestors.file import FileIngestor

# Directories that are always excluded from traversal.
_ALWAYS_EXCLUDE_DIRS: set[str] = {
    "__pycache__",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".eggs",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".hg",
    ".svn",
}


class DirectoryIngestor:
    """Recursively ingest all files in a directory tree.

    * Respects ``.gitignore`` patterns found at the directory root.
    * Skips hidden files/directories (names starting with ``"."``).
    * Skips common non-source directories (``__pycache__``, ``node_modules``, etc.).
    """

    def __init__(self) -> None:
        self._file_ingestor = FileIngestor()

    # ------------------------------------------------------------------
    # Ingestor interface
    # ------------------------------------------------------------------

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]:
        """Recursively read all files under *uri*."""
        root = Path(uri)
        if not root.is_dir():
            return []

        ignore_patterns = self._load_gitignore(root)
        results: list[RawContent] = []

        for file_path in self._walk(root, root, ignore_patterns):
            contents = self._file_ingestor.read(str(file_path), metadata)
            results.extend(contents)

        return results

    def chunk(self, contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
        """Chunk directory contents using token-window splitting."""
        return token_chunk(contents, config)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_gitignore(root: Path) -> list[str]:
        """Parse the root ``.gitignore`` and return a list of glob patterns."""
        gitignore = root / ".gitignore"
        if not gitignore.is_file():
            return []
        patterns: list[str] = []
        for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped)
        return patterns

    @staticmethod
    def _is_ignored(rel_path: str, patterns: list[str]) -> bool:
        """Return True if *rel_path* matches any gitignore-style pattern."""
        for pattern in patterns:
            # Match against the full relative path and just the basename.
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            if fnmatch.fnmatch(rel_path.split("/")[-1], pattern):
                return True
            # Handle directory patterns like "build/" by matching path segments.
            clean = pattern.rstrip("/")
            if fnmatch.fnmatch(rel_path, clean) or fnmatch.fnmatch(
                rel_path.split("/")[-1], clean
            ):
                return True
        return False

    @classmethod
    def _walk(
        cls,
        current: Path,
        root: Path,
        ignore_patterns: list[str],
    ) -> list[Path]:
        """Recursively yield file paths that should be ingested."""
        files: list[Path] = []
        try:
            entries = sorted(current.iterdir())
        except PermissionError:
            return files

        for entry in entries:
            name = entry.name
            # Skip hidden entries.
            if name.startswith("."):
                continue

            rel = entry.relative_to(root).as_posix()

            if entry.is_dir():
                if name in _ALWAYS_EXCLUDE_DIRS:
                    continue
                if cls._is_ignored(rel, ignore_patterns):
                    continue
                files.extend(cls._walk(entry, root, ignore_patterns))
            elif entry.is_file():
                if cls._is_ignored(rel, ignore_patterns):
                    continue
                files.append(entry)

        return files
