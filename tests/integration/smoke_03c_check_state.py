"""Read-only inspection of KB state after a smoke-ingest run.

Prints kb_info, every source row (status, chunk count, dedup key,
ingested_at, file basename), and a sample of the page list. Useful when
investigating why a previous tool call timed out or returned partial data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import resolve_kb_id  # noqa: E402
from mcp_client import KbClient  # noqa: E402


def main() -> None:
    kb_id = resolve_kb_id()
    with KbClient() as kb:
        info = kb.call("kb_info", kb_id=kb_id)
        print("kb_info:", info)
        print()
        srcs = kb.call("kb_list_sources", kb_id=kb_id)
        print(f"sources: {len(srcs or [])}")
        for s in srcs or []:
            print(
                f"  {s['status']:10s}  chunks={s['chunk_count']:3d}  "
                f"dedup={s.get('dedup_key')!r:25s}  "
                f"ingested_at={s.get('ingested_at')}  uri=...{Path(s['uri']).name}"
            )
        print()
        pages = kb.call("kb_list_pages", kb_id=kb_id) or []
        print(f"pages: {len(pages)}")
        for p in pages[:5]:
            keys = list(p.keys()) if isinstance(p, dict) else None
            print(f"  keys={keys}  sample={str(p)[:200]}")


if __name__ == "__main__":
    main()
