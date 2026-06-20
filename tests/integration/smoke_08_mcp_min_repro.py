"""Minimal MCP repro: a single kb_ingest call against a small file in a fresh server.

If this hangs >180s while the in-process equivalent (`smoke_04_direct_ingest`)
finishes in ~3s, the MCP `_with_tool_timeout` wrapper is broken — likely
deadlocking on the executor / lock.

Strategy:
  1. Pre-clean any prior failed source via in-process service (so the MCP
     call hits a clean state).
  2. Spawn a brand-new MCP server.
  3. Make exactly ONE kb_ingest tool call and time it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import ensure_src_on_path, require_env, resolve_kb_id, transcripts_dir  # noqa: E402
from mcp_client import KbClient  # noqa: E402

require_env("AGENT_KB_SAVES_DIR", "set it to your persistent KB saves directory.")
ensure_src_on_path()

from agent_knowledgebase.config import Settings  # noqa: E402
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService  # noqa: E402

# rJCgvnXgOiU is already ingested under `yt:rJCgvnXgOiU`; we use a distinct
# dedup_key so the repro is independent of the stable corpus.
TARGET_VID = "rJCgvnXgOiU"
TEST_DEDUP_SUFFIX = "__mcp_repro"


def main() -> None:
    kb_id = resolve_kb_id()
    target = next(p for p in transcripts_dir().glob(f"*{TARGET_VID}*.md"))
    print(f"file: {target.name}  size={target.stat().st_size}")

    test_dedup = f"yt:{TARGET_VID}{TEST_DEDUP_SUFFIX}"

    # Pre-clean: drop any source for our test dedup_key.
    svc = KnowledgebaseService(Settings())
    for s in svc.list_sources(kb_id):
        if s.dedup_key == test_dedup:
            print(f"  pre-cleaning source {s.id}")
            svc.remove_source(s.id)

    print("\n--- Spawn fresh MCP server ---")
    with KbClient() as kb:
        print("--- Single kb_ingest tool call ---")
        t0 = time.monotonic()
        try:
            result = kb.call(
                "kb_ingest",
                kb_id=kb_id,
                source_type="file",
                uri=str(target),
                metadata=json.dumps({"video_id": TARGET_VID, "label": "mcp_repro"}),
                dedup_key=test_dedup,
                dedup_policy="force-add",
            )
            elapsed = time.monotonic() - t0
            print(f"DONE in {elapsed:.1f}s -> {json.dumps(result)[:200]}")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"EXC after {elapsed:.1f}s: {exc}")

        print("\n--- STDERR_TAIL ---")
        print(kb.stderr_tail(2500))


if __name__ == "__main__":
    main()
