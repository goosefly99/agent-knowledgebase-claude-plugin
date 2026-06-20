"""Print the FULL content of top vector hits for a query.

Used to verify retrieval quality from an agent's perspective — the truncated
previews from `smoke_10` are enough to spot good/bad ranking but not enough
to judge whether each hit's content is actually answer-relevant.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import ensure_src_on_path, require_env, resolve_kb_id  # noqa: E402

require_env("AGENT_KB_SAVES_DIR", "set it to your persistent KB saves directory.")
ensure_src_on_path()

from agent_knowledgebase.config import Settings  # noqa: E402
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService  # noqa: E402


def main() -> None:
    kb_id = resolve_kb_id()
    query = os.environ.get("AGENT_KB_TEST_QUERY", "claude obsidian")
    top_k = int(os.environ.get("AGENT_KB_TEST_TOP_K", "3"))

    settings = Settings()
    svc = KnowledgebaseService(settings)
    results = svc.query(kb_id, query, top_k=top_k)
    for i, r in enumerate(results, 1):
        metadata = r.metadata or {}
        vid = metadata.get("video_id") or metadata.get("file_path", "")
        score = r.score
        print(f"\n========== Hit {i} (score={score:.4f}, video={vid}) ==========")
        print(r.content)
    print("\n--- end ---")


if __name__ == "__main__":
    main()
