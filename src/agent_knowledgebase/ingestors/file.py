"""FileIngestor -- read individual files (text, markdown, JSON, PDF)."""

from __future__ import annotations

import json
from pathlib import Path

from agent_knowledgebase.ingestors import ChunkConfig, Chunk, RawContent, token_chunk


class FileIngestor:
    """Ingest a single file from the local filesystem.

    Supported formats:
    - ``.pdf`` -- uses *pdfplumber* to extract text.
    - ``.json`` -- pretty-prints the JSON content.
    - Everything else (`.md`, `.txt`, `.py`, `.js`, `.ts`, ...) -- read as
      plain text.
    """

    # ------------------------------------------------------------------
    # Ingestor interface
    # ------------------------------------------------------------------

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]:
        """Read a single file and return its content."""
        path = Path(uri)
        extension = path.suffix.lower()
        file_metadata = {
            "file_path": str(path),
            "file_type": extension,
            **(metadata or {}),
        }

        text = self._read_by_type(path, extension)
        if not text:
            return []
        return [RawContent(text=text, metadata=file_metadata)]

    def chunk(self, contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
        """Chunk file contents using token-window splitting."""
        return token_chunk(contents, config)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read_by_type(path: Path, extension: str) -> str:
        """Dispatch reading strategy based on file extension."""
        if extension == ".pdf":
            return FileIngestor._read_pdf(path)
        if extension == ".json":
            return FileIngestor._read_json(path)
        return FileIngestor._read_text(path)

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_json(path: Path) -> str:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return raw

    @staticmethod
    def _read_pdf(path: Path) -> str:
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
