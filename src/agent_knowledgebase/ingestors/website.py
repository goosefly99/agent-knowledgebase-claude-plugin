"""WebsiteIngestor -- extract main content from a web page via trafilatura."""

from __future__ import annotations

from agent_knowledgebase.ingestors import ChunkConfig, Chunk, RawContent, token_chunk


class WebsiteIngestor:
    """Fetch a URL and extract its main textual content.

    Uses the *trafilatura* library for downloading and content extraction.
    """

    # ------------------------------------------------------------------
    # Ingestor interface
    # ------------------------------------------------------------------

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]:
        """Fetch *uri* and extract text with trafilatura."""
        import trafilatura

        downloaded = trafilatura.fetch_url(uri)
        if downloaded is None:
            return []

        text = trafilatura.extract(downloaded) or ""
        if not text:
            return []

        title = trafilatura.extract(downloaded, output_format="xml") or ""
        # Attempt to pull a <title> tag from the XML output.
        extracted_title = _extract_title(title)

        page_metadata: dict = {
            "url": uri,
            "title": extracted_title,
            **(metadata or {}),
        }
        return [RawContent(text=text, metadata=page_metadata)]

    def chunk(self, contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
        """Chunk website contents using token-window splitting."""
        return token_chunk(contents, config)


def _extract_title(xml_output: str) -> str:
    """Best-effort title extraction from trafilatura XML output."""
    import re

    match = re.search(r"title=[\"']([^\"']+)[\"']", xml_output)
    if match:
        return match.group(1)
    return ""
