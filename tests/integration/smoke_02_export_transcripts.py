"""Step 2: regenerate the fixture transcript corpus from a YouTube cache DB.

Optional. Only needed when refreshing the bundled fixtures in
`fixtures/transcripts/`. Reads the youtube-mcp cache sqlite at
`AGENT_KB_TEST_YT_DB`, exports each candidate video as a markdown file with
YAML front-matter, and writes a `transcripts_manifest.json` whose `path`
fields are stored relative to the manifest's own directory (so the corpus is
relocatable).

Schema expected at `AGENT_KB_TEST_YT_DB`:
  videos(video_id PK, title, channel_title, published_at, duration, view_count, ...)
  transcripts(video_id PK, full_text, ...)
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import resolve_yt_db, transcripts_dir, transcripts_manifest_path  # noqa: E402

CANDIDATES = [
    "OSZdFnQmgRw", "QHlB-RJfx8w", "pAIF7vZm5k0", "rJCgvnXgOiU",
    "TXcr0x9SIXA", "UHVFcUzAGlM", "Ek0tZootK00", "ZIS_okcwQ-Q",
    "yfeHoOkn2TI", "9d5bzxVsocw", "Y2rpFa43jTo", "yJK5GueSHmU",
    "sboNwYmH3AY", "pIoKdpsuODg", "yF9kGESAi3M",
]


def slugify(s: str) -> str:
    out: list[str] = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("-")
    return "".join(out).strip("-")[:60] or "untitled"


def main() -> None:
    yt_db = resolve_yt_db()
    out_dir = transcripts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(yt_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    written: list[dict] = []
    for vid in CANDIDATES:
        v = cur.execute("SELECT * FROM videos WHERE video_id=?", (vid,)).fetchone()
        t = cur.execute("SELECT * FROM transcripts WHERE video_id=?", (vid,)).fetchone()
        if not v:
            print(f"  SKIP {vid}: no video row")
            continue
        title = v["title"] or vid
        channel = v["channel_title"] or ""
        url = f"https://www.youtube.com/watch?v={vid}"
        body = (t["full_text"] if t else "") or ""

        slug = f"{slugify(title)}__{vid}"
        out = out_dir / f"{slug}.md"
        meta_lines = [
            "---",
            f'video_id: "{vid}"',
            f'title: {json.dumps(title)}',
            f'channel: {json.dumps(channel)}',
            f'url: "{url}"',
            f'published_at: "{v["published_at"] or ""}"',
            f'duration: "{v["duration"] or ""}"',
            f'view_count: {v["view_count"] or 0}',
            "---",
            "",
            f"# {title}",
            "",
            f"**Channel:** {channel}  ",
            f"**URL:** {url}  ",
            "",
            "## Transcript",
            "",
            body if body.strip() else "_(no transcript available)_",
        ]
        out.write_text("\n".join(meta_lines), encoding="utf-8")
        rel = out.relative_to(transcripts_manifest_path().parent).as_posix()
        written.append({"video_id": vid, "path": rel, "size": out.stat().st_size})
        print(f"  WROTE {vid} -> {out.name} ({out.stat().st_size} bytes)")

    print(f"\n=== WROTE {len(written)} files to {out_dir} ===")
    manifest = transcripts_manifest_path()
    manifest.write_text(json.dumps(written, indent=2), encoding="utf-8")
    print(f"=== Manifest: {manifest} ===")


if __name__ == "__main__":
    main()
