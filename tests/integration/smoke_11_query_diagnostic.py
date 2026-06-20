"""Two-call kb_query (warm-cache test) + in-process kb_search reproduction.

Goals:
  1. Confirm whether kb_query's 180s first-call timeout is the same cold-load
     issue from the ingest deadlock (chromadb collection load) or a regression
     specific to retrieval.
  2. Capture the FULL traceback for the 'RustBindingsAPI' kb_search error by
     running the same code path in-process (so we get a real Python traceback,
     not just a stringified MCP error).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import ensure_src_on_path, require_env, resolve_kb_id  # noqa: E402
from mcp_client import KbClient  # noqa: E402

require_env("AGENT_KB_SAVES_DIR", "set it to your persistent KB saves directory.")
ensure_src_on_path()

QUERY = "claude obsidian"


def fmt_hit(idx: int, hit: dict) -> str:
    score = hit.get("score")
    score_s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    metadata = hit.get("metadata") or {}
    vid = metadata.get("video_id") or metadata.get("file_path") or ""
    chunk = hit.get("chunk_id") or hit.get("source_id") or ""
    content = (hit.get("content") or "").replace("\n", " ").strip()
    if len(content) > 160:
        content = content[:160] + "..."
    return (
        f"  [{idx}] score={score_s:>8}  vid={str(vid):>12}  "
        f"src={str(chunk)[:36]}  preview={content!r}"
    )


def part_a_two_calls(kb_id: str) -> None:
    print("=" * 70)
    print("PART A — two kb_query calls in same MCP session")
    print("=" * 70)
    with KbClient() as kb:
        for label in ("call #1", "call #2"):
            print(f"\n--- {label}: kb_query text={QUERY!r} top_k=5 ---")
            t0 = time.time()
            try:
                result = kb.call("kb_query", kb_id=kb_id, text=QUERY, top_k=5)
                elapsed = time.time() - t0
                print(f"  elapsed: {elapsed:.2f}s")
                if isinstance(result, dict) and "error" in result:
                    print(f"  embedder error payload: {result}")
                elif isinstance(result, list):
                    print(f"  hits: {len(result)}")
                    for i, h in enumerate(result, 1):
                        print(fmt_hit(i, h))
                else:
                    print(f"  raw: {result!r}")
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  elapsed: {elapsed:.2f}s  EXC: {type(exc).__name__}: {exc}")
        # After both calls, dump server stderr.
        print("\n--- server stderr (last 2000 chars) ---")
        print(kb.stderr_tail(2000))


def part_b_inprocess_search(kb_id: str) -> None:
    print()
    print("=" * 70)
    print("PART B — in-process reproduction of kb_search to get traceback")
    print("=" * 70)
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService

    settings = Settings()
    svc = KnowledgebaseService(settings)
    print(f"\nkb_search(kb_id={kb_id!r}, text={QUERY!r})")
    t0 = time.time()
    try:
        results = svc.search(kb_id, QUERY, top_k=8)
        elapsed = time.time() - t0
        print(f"  elapsed: {elapsed:.2f}s")
        print(f"  hits: {len(results)}")
        for i, r in enumerate(results, 1):
            preview = (getattr(r, "content", "") or "")[:160].replace("\n", " ")
            metadata = getattr(r, "metadata", {}) or {}
            score = getattr(r, "score", None)
            print(
                f"  [{i}] score={score!s:>8} title={metadata.get('title','')!r:>40} "
                f"preview={preview!r}"
            )
    except BaseException:
        elapsed = time.time() - t0
        print(f"  elapsed: {elapsed:.2f}s")
        print("  --- FULL TRACEBACK ---")
        traceback.print_exc()


def main() -> None:
    kb_id = resolve_kb_id()
    part_a_two_calls(kb_id)
    part_b_inprocess_search(kb_id)


if __name__ == "__main__":
    main()
