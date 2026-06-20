"""Isolate the kb_search 'RustBindingsAPI' error.

Hypothesis: kb_search's MCP failure is a side-effect of the prior kb_query
timeout (which tears down the worker thread mid-chromadb-load, leaving the
chromadb rust bindings handle in a bad state for the next caller).

Test: start a FRESH MCP session and call kb_search FIRST (no kb_query
before it). If it succeeds, the bug is the timeout-reset side-effect. If
it fails the same way, kb_search has its own bug.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import resolve_kb_id  # noqa: E402
from mcp_client import KbClient  # noqa: E402

QUERY = "claude obsidian"


def main() -> None:
    kb_id = resolve_kb_id()
    print("=== Test A: kb_search FIRST (no prior kb_query) ===\n")
    with KbClient() as kb:
        t0 = time.time()
        try:
            result = kb.call("kb_search", kb_id=kb_id, text=QUERY, top_k=8)
            elapsed = time.time() - t0
            print(f"  elapsed: {elapsed:.2f}s")
            if isinstance(result, list):
                print(f"  hits: {len(result)}")
                for i, h in enumerate(result[:5], 1):
                    metadata = h.get("metadata") or {}
                    print(f"  [{i}] score={h.get('score')} title={metadata.get('title','')!r}")
            else:
                print(f"  result: {result!r}")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  elapsed: {elapsed:.2f}s")
            print(f"  EXC: {type(exc).__name__}: {exc}")
        print("\n  --- server stderr (last 1500 chars) ---")
        print(kb.stderr_tail(1500))

    print("\n=== Test B: kb_query (cold) -> kb_search (after timeout) ===\n")
    with KbClient() as kb:
        print("  -- kb_query (will likely time out) --")
        t0 = time.time()
        try:
            r = kb.call("kb_query", kb_id=kb_id, text=QUERY, top_k=5)
            print(f"    elapsed: {time.time()-t0:.2f}s  result_type: {type(r).__name__}")
            if isinstance(r, dict) and "error" in r:
                print(f"    payload: {r}")
        except Exception as exc:
            print(f"    elapsed: {time.time()-t0:.2f}s  EXC: {exc}")
        print("\n  -- kb_search (after timeout) --")
        t0 = time.time()
        try:
            result = kb.call("kb_search", kb_id=kb_id, text=QUERY, top_k=8)
            elapsed = time.time() - t0
            print(f"    elapsed: {elapsed:.2f}s")
            if isinstance(result, list):
                print(f"    hits: {len(result)}")
            else:
                print(f"    result: {result!r}")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"    elapsed: {elapsed:.2f}s")
            print(f"    EXC: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
