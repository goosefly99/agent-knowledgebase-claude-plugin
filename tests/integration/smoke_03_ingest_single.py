"""Step 3a: smoke-test ingest with the smallest fixture transcript via MCP."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import resolve_kb_id, transcripts_dir  # noqa: E402
from mcp_client import KbClient  # noqa: E402

# yJK5GueSHmU is the smallest fixture transcript (~1.2 KB).
SMALLEST_VID = "yJK5GueSHmU"


def main() -> None:
    kb_id = resolve_kb_id()
    target = next(p for p in transcripts_dir().glob(f"*{SMALLEST_VID}*.md"))
    print(f"target: {target}  size={target.stat().st_size}")

    with KbClient() as kb:
        print("\n=== kb_ingest (single file, smallest) ===")
        result = kb.call(
            "kb_ingest",
            kb_id=kb_id,
            source_type="file",
            uri=str(target),
            metadata=json.dumps({
                "video_id": SMALLEST_VID,
                "kb_source_label": f"yt-transcript:{SMALLEST_VID}",
            }),
            dedup_key=f"yt:{SMALLEST_VID}",
            dedup_policy="skip",
        )
        print(json.dumps(result, indent=2, default=str))

        print("\n=== kb_list_sources ===")
        srcs = kb.call("kb_list_sources", kb_id=kb_id)
        print(json.dumps(srcs, indent=2, default=str))

        print("\n=== kb_list_pages ===")
        pages = kb.call("kb_list_pages", kb_id=kb_id)
        print(json.dumps(pages, indent=2, default=str)[:2000])

        print("\n=== STDERR_TAIL ===")
        print(kb.stderr_tail(2500))


if __name__ == "__main__":
    main()
