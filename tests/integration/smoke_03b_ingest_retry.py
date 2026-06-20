"""Retry the smoke-test ingest after Ollama warmup. Replace failed source first."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import resolve_kb_id, transcripts_dir  # noqa: E402
from mcp_client import KbClient  # noqa: E402

SMALLEST_VID = "yJK5GueSHmU"


def main() -> None:
    kb_id = resolve_kb_id()
    target = next(p for p in transcripts_dir().glob(f"*{SMALLEST_VID}*.md"))
    dedup_key = f"yt:{SMALLEST_VID}"

    with KbClient() as kb:
        # Find and remove any failed source for this video.
        srcs = kb.call("kb_list_sources", kb_id=kb_id) or []
        for s in srcs:
            if s.get("dedup_key") == dedup_key and s.get("status") == "failed":
                print(f"removing failed source {s['id']}")
                kb.call("kb_remove_source", source_id=s["id"])

        print("\n=== kb_ingest (force-add) ===")
        result = kb.call(
            "kb_ingest",
            kb_id=kb_id,
            source_type="file",
            uri=str(target),
            metadata=json.dumps({
                "video_id": SMALLEST_VID,
                "kb_source_label": f"yt-transcript:{SMALLEST_VID}",
            }),
            dedup_key=dedup_key,
            dedup_policy="force-add",
        )
        print(json.dumps(result, indent=2, default=str))

        print("\n=== kb_list_sources (after) ===")
        srcs2 = kb.call("kb_list_sources", kb_id=kb_id)
        for s in srcs2 or []:
            print(
                f"  {s['status']:8s}  chunks={s['chunk_count']}  "
                f"dedup={s.get('dedup_key')}  uri={Path(s['uri']).name}"
            )

        print("\n=== kb_list_pages (after) ===")
        pages = kb.call("kb_list_pages", kb_id=kb_id)
        print(f"page_count={len(pages or [])}")
        for p in (pages or [])[:3]:
            print(f"  {p}")


if __name__ == "__main__":
    main()
