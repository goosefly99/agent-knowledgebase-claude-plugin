"""Bulk-ingest all bundled fixture transcripts into the test KB in-process.

We bypass the MCP wrapper because of a confirmed `_with_tool_timeout`
deadlock (reproducibly hangs >180s post-embed even though direct in-process
ingest of the same file completes in 3-15s). See `smoke_08_mcp_min_repro`
for the minimal MCP repro.

Strategy:
  - For each manifest entry, call svc.ingest_source(file, ..., dedup_policy=force-add)
  - On dedup-key collision with a prior `failed` source, remove it first
  - Skip entries whose dedup_key is already `ingested`
  - Report per-file (status, chunks, elapsed) and write a summary JSON next
    to the manifest (ignored by git).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import (  # noqa: E402
    ensure_src_on_path,
    require_env,
    resolve_kb_id,
    transcripts_manifest_path,
)

require_env("AGENT_KB_SAVES_DIR", "set it to your persistent KB saves directory.")
ensure_src_on_path()

logging.basicConfig(level=logging.WARNING)

from agent_knowledgebase.config import Settings  # noqa: E402
from agent_knowledgebase.models import SourceType  # noqa: E402
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService  # noqa: E402


def main() -> None:
    kb_id = resolve_kb_id()
    settings = Settings()
    svc = KnowledgebaseService(settings)

    manifest_path = transcripts_manifest_path()
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"=== {len(items)} files to ingest ===\n")

    # Map existing sources by dedup_key for fast dedup-bookkeeping.
    existing = {s.dedup_key: s for s in svc.list_sources(kb_id) if s.dedup_key}

    summary: list[dict] = []
    overall_start = time.monotonic()
    for it in items:
        vid = it["video_id"]
        # Manifest paths are relative to the manifest's own directory.
        path = (manifest_path.parent / it["path"]).resolve()
        dedup_key = f"yt:{vid}"
        size = it["size"]

        prior = existing.get(dedup_key)
        if prior and prior.status.value == "ingested":
            print(f"  SKIP   {vid} ({size:>6d} B) - already ingested ({prior.chunk_count} chunks)")
            summary.append({
                "video_id": vid,
                "status": "already_ingested",
                "chunks": prior.chunk_count,
                "elapsed_s": 0.0,
            })
            continue
        if prior and prior.status.value == "failed":
            svc.remove_source(prior.id)

        t0 = time.monotonic()
        try:
            result = svc.ingest_source(
                kb_id,
                SourceType.file,
                str(path),
                metadata={"video_id": vid, "kb_source_label": f"yt-transcript:{vid}"},
                dedup_key=dedup_key,
                dedup_policy="force-add",
            )
            elapsed = time.monotonic() - t0
            print(f"  OK     {vid} ({size:>6d} B) -> {result.chunk_count:>3d} chunks  {elapsed:6.2f}s")
            summary.append({
                "video_id": vid,
                "status": result.status.value,
                "chunks": result.chunk_count,
                "elapsed_s": round(elapsed, 2),
                "source_id": result.id,
            })
        except BaseException as exc:
            elapsed = time.monotonic() - t0
            print(
                f"  FAIL   {vid} ({size:>6d} B)  {elapsed:6.2f}s  "
                f"{type(exc).__name__}: {str(exc)[:120]}"
            )
            summary.append({
                "video_id": vid,
                "status": "exception",
                "elapsed_s": round(elapsed, 2),
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            })

    total = time.monotonic() - overall_start
    ok = sum(1 for s in summary if s["status"] in ("ingested", "already_ingested"))
    print(f"\n=== {ok}/{len(items)} ingested in {total:.1f}s ===")

    # Summary is per-run output; not committed to the repo.
    out = manifest_path.parent / "ingest_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"=== summary: {out} ===")


if __name__ == "__main__":
    main()
