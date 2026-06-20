"""Query the test KB with 'claude obsidian' via the MCP server.

Tests both vector (kb_query) and keyword (kb_search) paths and reports
hit-quality so we can tell whether retrieval picks up the intended videos
(Karpathy Obsidian-RAG videos, Obsidian Second Brain videos).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import resolve_kb_id  # noqa: E402
from mcp_client import KbClient  # noqa: E402

QUERY = "claude obsidian"
TOP_K = 8


def fmt_hit(idx: int, hit: dict) -> str:
    score = hit.get("score")
    score_s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    metadata = hit.get("metadata") or {}
    vid = metadata.get("video_id") or metadata.get("file_path") or ""
    chunk = hit.get("chunk_id") or hit.get("id") or ""
    content = (hit.get("content") or "").replace("\n", " ").strip()
    if len(content) > 180:
        content = content[:180] + "..."
    return (
        f"  [{idx}] score={score_s:>8}  vid={str(vid):>12}  "
        f"chunk={str(chunk)[:8]}  preview={content!r}"
    )


def main() -> None:
    kb_id = resolve_kb_id()
    with KbClient() as kb:
        # Sanity: confirm KB exists and looks healthy.
        info_raw = kb.call("kb_info", kb_id=kb_id)
        info = info_raw if isinstance(info_raw, dict) else json.loads(info_raw)
        print(
            f"=== KB {info.get('name')} ({info.get('source_count')} sources, "
            f"{info.get('page_count')} pages, "
            f"embedder={info.get('embedder_model','?')}) ===\n"
        )

        # --- vector retrieval ---
        print(f"--- kb_query (vector) text={QUERY!r} top_k={TOP_K} ---")
        t0 = time.time()
        try:
            q_raw = kb.call("kb_query", kb_id=kb_id, text=QUERY, top_k=TOP_K)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            print("  stderr_tail:")
            print(kb.stderr_tail(1500))
            q_raw = None
        elapsed_q = time.time() - t0
        print(f"  elapsed: {elapsed_q:.2f}s")
        q_hits: list = []
        if q_raw is not None:
            q_hits = q_raw if isinstance(q_raw, list) else (
                json.loads(q_raw) if isinstance(q_raw, str) else []
            )
            if isinstance(q_hits, dict) and "error" in q_hits:
                print(f"  embedder error: {q_hits}")
                q_hits = []
            print(f"  hits: {len(q_hits)}")
            for i, h in enumerate(q_hits, 1):
                print(fmt_hit(i, h))
        print()

        # --- keyword search ---
        print(f"--- kb_search (keyword) text={QUERY!r} top_k={TOP_K} ---")
        t0 = time.time()
        try:
            s_raw = kb.call("kb_search", kb_id=kb_id, text=QUERY, top_k=TOP_K)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            s_raw = None
        elapsed_s = time.time() - t0
        print(f"  elapsed: {elapsed_s:.2f}s")
        if s_raw is not None:
            s_hits = s_raw if isinstance(s_raw, list) else (
                json.loads(s_raw) if isinstance(s_raw, str) else []
            )
            print(f"  hits: {len(s_hits)}")
            for i, h in enumerate(s_hits, 1):
                print(fmt_hit(i, h))
        print()

        # Summary tally — videos by source_id (vector path).
        if q_hits:
            tally: dict[str, int] = {}
            for h in q_hits:
                metadata = h.get("metadata") or {}
                vid = metadata.get("video_id") or metadata.get("file_path") or "?"
                tally[vid] = tally.get(vid, 0) + 1
            print("--- vector hit tally by video_id ---")
            for vid, n in sorted(tally.items(), key=lambda kv: -kv[1]):
                print(f"  {vid:>16}: {n}")


if __name__ == "__main__":
    main()
