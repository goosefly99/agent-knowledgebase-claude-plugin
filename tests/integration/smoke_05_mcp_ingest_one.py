"""Re-attempt one MCP ingest with timing, against a fresh transcript file."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import resolve_kb_id, transcripts_dir  # noqa: E402
from mcp_client import KbClient  # noqa: E402

# pIoKdpsuODg is ~9.5 KB and was never used in earlier smoke runs, so it
# acts as a "clean" target for measuring single-call MCP ingest latency.
TARGET_VID = "pIoKdpsuODg"


def main() -> None:
    kb_id = resolve_kb_id()
    target = next(p for p in transcripts_dir().glob(f"*{TARGET_VID}*.md"))
    print(f"target: {target.name}  size={target.stat().st_size}")

    with KbClient() as kb:
        print("\n[1] kb_list_sources")
        t0 = time.monotonic()
        srcs = kb.call("kb_list_sources", kb_id=kb_id)
        print(f"    {time.monotonic()-t0:.2f}s; count={len(srcs or [])}")

        print("\n[2] kb_ingest (force-add)")
        t0 = time.monotonic()
        try:
            result = kb.call(
                "kb_ingest",
                kb_id=kb_id,
                source_type="file",
                uri=str(target),
                metadata=json.dumps({"video_id": TARGET_VID}),
                dedup_key=f"yt:{TARGET_VID}",
                dedup_policy="force-add",
            )
            print(f"    {time.monotonic()-t0:.2f}s; result={json.dumps(result)[:300]}")
        except Exception as exc:
            print(f"    {time.monotonic()-t0:.2f}s  EXCEPTION: {exc}")

        print("\n[3] STDERR_TAIL")
        print(kb.stderr_tail(2500))


if __name__ == "__main__":
    main()
