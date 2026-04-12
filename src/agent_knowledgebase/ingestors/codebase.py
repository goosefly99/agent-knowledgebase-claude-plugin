"""CodebaseIngestor -- language-aware code chunking."""

from __future__ import annotations

import re
from pathlib import Path

import tiktoken

from agent_knowledgebase.ingestors import ChunkConfig, Chunk, RawContent, token_chunk
from agent_knowledgebase.ingestors.directory import DirectoryIngestor

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Map file extensions to language identifiers.
_EXTENSION_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "shell",
    ".bash": "shell",
}

# Regex patterns for detecting top-level symbols by language.
# Each pattern MUST use a named group ``name`` for the symbol identifier.
_SYMBOL_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "python": [
        ("class", re.compile(r"^class\s+(?P<name>\w+)", re.MULTILINE)),
        ("function", re.compile(r"^def\s+(?P<name>\w+)", re.MULTILINE)),
    ],
    "javascript": [
        ("class", re.compile(r"^class\s+(?P<name>\w+)", re.MULTILINE)),
        (
            "function",
            re.compile(
                r"^(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)",
                re.MULTILINE,
            ),
        ),
    ],
    "typescript": [
        ("class", re.compile(r"^(?:export\s+)?class\s+(?P<name>\w+)", re.MULTILINE)),
        (
            "function",
            re.compile(
                r"^(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)",
                re.MULTILINE,
            ),
        ),
    ],
}


class CodebaseIngestor:
    """Ingest a code repository with language-aware chunking.

    Uses regex-based symbol detection for Python, JavaScript, and TypeScript
    to split on function/class boundaries.  Falls back to token-window
    chunking for other languages.
    """

    def __init__(self) -> None:
        self._dir_ingestor = DirectoryIngestor()

    # ------------------------------------------------------------------
    # Ingestor interface
    # ------------------------------------------------------------------

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]:
        """Read a codebase directory and annotate with language metadata."""
        raw_contents = self._dir_ingestor.read(uri, metadata)
        for raw in raw_contents:
            file_path = raw.metadata.get("file_path", "")
            ext = Path(file_path).suffix.lower()
            raw.metadata["language"] = _EXTENSION_LANGUAGE.get(ext, "unknown")
        return raw_contents

    def chunk(self, contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
        """Chunk with language-aware splitting where possible.

        For supported languages the text is first split on function/class
        boundaries.  Each symbol block that exceeds ``config.chunk_size``
        tokens is further subdivided with token-window chunking.

        For unsupported languages the standard token-window chunker is used.
        """
        chunks: list[Chunk] = []

        for raw in contents:
            language = raw.metadata.get("language", "unknown")
            patterns = _SYMBOL_PATTERNS.get(language)

            if patterns:
                chunks.extend(self._language_chunk(raw, patterns, config))
            else:
                chunks.extend(token_chunk([raw], config))

        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _language_chunk(
        raw: RawContent,
        patterns: list[tuple[str, re.Pattern[str]]],
        config: ChunkConfig,
    ) -> list[Chunk]:
        """Split *raw* on regex-detected symbol boundaries."""
        text = raw.text
        # Collect all match positions with their metadata.
        boundaries: list[tuple[int, str, str]] = []  # (pos, symbol_type, symbol_name)
        for symbol_type, pattern in patterns:
            for m in pattern.finditer(text):
                boundaries.append((m.start(), symbol_type, m.group("name")))

        # Sort by position in file.
        boundaries.sort(key=lambda b: b[0])

        if not boundaries:
            # No detected symbols -- fall back to token chunking.
            return list(token_chunk([raw], config))

        chunks: list[Chunk] = []

        # Handle text before the first symbol (module-level code).
        if boundaries[0][0] > 0:
            preamble = text[: boundaries[0][0]]
            if preamble.strip():
                chunks.extend(
                    _make_symbol_chunks(
                        preamble,
                        {**raw.metadata, "symbol_type": "module"},
                        config,
                    )
                )

        for i, (pos, sym_type, sym_name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            segment = text[pos:end]
            if not segment.strip():
                continue
            meta = {
                **raw.metadata,
                "symbol_type": sym_type,
                "symbol_name": sym_name,
            }
            chunks.extend(_make_symbol_chunks(segment, meta, config))

        return chunks


def _make_symbol_chunks(
    text: str,
    metadata: dict,
    config: ChunkConfig,
) -> list[Chunk]:
    """Create chunk(s) for a single symbol block.

    If the block fits within ``config.chunk_size`` tokens it becomes a
    single chunk.  Otherwise, token-window splitting is applied.
    """
    tokens = _ENCODING.encode(text)
    if len(tokens) <= config.chunk_size:
        return [
            Chunk(source_id="", kb_id="", content=text, metadata=dict(metadata))
        ]
    # Oversized symbol -- use standard token chunking.
    return list(
        token_chunk(
            [RawContent(text=text, metadata=metadata)],
            config,
        )
    )
