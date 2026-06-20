"""Two sequential MCP ingest calls in one session.

Hypothesis: chromadb HNSW index cold-load on the first add to a populated
collection takes ~minutes. If true, call #1 hangs but call #2 is fast
because the index is now warm in process memory.
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

TARGETS = ["pAIF7vZm5k0", "9d5bzxVsocw"]
DEDUP_SUFFIX = "__mcp9"


def main() -> None:
    kb_id = resolve_kb_id()
    svc = KnowledgebaseService(Settings())
    for vid in TARGETS:
        for s in svc.list_sources(kb_id):
            if s.dedup_key == f"yt:{vid}{DEDUP_SUFFIX}":
                svc.remove_source(s.id)

    files = [next(transcripts_dir().glob(f"*{vid}*.md")) for vid in TARGETS]

    with KbClient() as kb:
        # Tiny call to confirm the server is responsive.
        kb.call("kb_list")
        for vid, path in zip(TARGETS, files):
            print(f"\n=== ingest {vid}  ({path.stat().st_size} B) ===")
            t0 = time.monotonic()
            try:
                # mcp_client uses 600s recv timeout; the server's tool wrapper
                # is still 180s — we want to see actual work duration.
                result = kb.call(
                    "kb_ingest",
                    kb_id=kb_id,
                    source_type="file",
                    uri=str(path),
                    metadata=json.dumps({"video_id": vid}),
                    dedup_key=f"yt:{vid}{DEDUP_SUFFIX}",
                    dedup_policy="force-add",
                )
                elapsed = time.monotonic() - t0
                print(f"  done {elapsed:.1f}s  result={json.dumps(result)[:200]}")
            except Exception as exc:
                elapsed = time.monotonic() - t0
                print(f"  EXC {elapsed:.1f}s  {exc}")


if __name__ == "__main__":
    main()
