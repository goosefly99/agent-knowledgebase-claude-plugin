"""MCP server entry point for agent-knowledgebase."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "agent-knowledgebase",
    description="Personal knowledgebase plugin — ingest sources, build wiki artifacts, query with vector search",
)


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
