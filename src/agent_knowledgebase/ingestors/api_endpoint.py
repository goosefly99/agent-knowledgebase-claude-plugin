"""ApiEndpointIngestor -- fetch content from REST API endpoints."""

from __future__ import annotations

import json

from agent_knowledgebase.ingestors import ChunkConfig, Chunk, RawContent, token_chunk


class ApiEndpointIngestor:
    """Ingest content from an HTTP REST API endpoint.

    Performs a GET request using *httpx* and parses the response.
    JSON responses are pretty-printed; other content types are returned
    as plain text.
    """

    # ------------------------------------------------------------------
    # Ingestor interface
    # ------------------------------------------------------------------

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]:
        """Fetch *uri* via GET and return the response body."""
        import httpx

        meta = metadata or {}
        headers: dict = meta.get("headers", {})

        response = httpx.get(uri, headers=headers, follow_redirects=True, timeout=30.0)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                data = response.json()
                text = json.dumps(data, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                text = response.text
        else:
            text = response.text

        if not text:
            return []

        result_metadata: dict = {
            "url": uri,
            "status_code": response.status_code,
            "content_type": content_type,
            **(meta),
        }
        return [RawContent(text=text, metadata=result_metadata)]

    def chunk(self, contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
        """Chunk API content using token-window splitting."""
        return token_chunk(contents, config)
