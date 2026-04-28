"""Query the test KB to verify end-to-end retrieval works (in-process)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import ensure_src_on_path, require_env, resolve_kb_id  # noqa: E402

require_env("AGENT_KB_SAVES_DIR", "set it to your persistent KB saves directory.")
ensure_src_on_path()

from agent_knowledgebase.config import Settings  # noqa: E402
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService  # noqa: E402

QUERIES = [
    "agentic RAG architecture",
    "LightRAG knowledge graph extraction",
    "vector database for Claude Code memory",
    "Karpathy second brain Obsidian",
    "REFRAG embedding compression",
]


def main() -> None:
    kb_id = resolve_kb_id()
    settings = Settings()
    svc = KnowledgebaseService(settings)
    info = svc.get_kb(kb_id)
    print(f"=== KB: {info.name} ({info.page_count} pages, {info.source_count} sources) ===\n")

    for q in QUERIES:
        print(f"--- query: {q!r} ---")
        try:
            results = svc.query(kb_id, q, top_k=3)
        except BaseException as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            print()
            continue
        for i, r in enumerate(results, 1):
            content = (getattr(r, "content", "") or "")[:140].replace("\n", " ")
            score = getattr(r, "score", None)
            metadata = getattr(r, "metadata", {}) or {}
            vid = metadata.get("video_id") or metadata.get("file_path", "")
            print(f"  [{i}] score={score!s:>10}  vid={vid!s:>20}  preview={content!r}")
        if not results:
            print("  (no hits)")
        print()


if __name__ == "__main__":
    main()
