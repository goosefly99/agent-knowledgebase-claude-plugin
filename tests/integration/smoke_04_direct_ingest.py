"""Bypass MCP — call the ingestion service in-process and time every phase.

Control experiment for the 180s MCP cold-load deadlock: confirms that the
underlying `KnowledgebaseService.ingest_source` call completes in seconds
when invoked directly, isolating the bug to the MCP `_with_tool_timeout`
wrapper rather than the ingestion code itself.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import ensure_src_on_path, require_env, resolve_kb_id, transcripts_dir  # noqa: E402

require_env("AGENT_KB_SAVES_DIR", "set it to your persistent KB saves directory.")
ensure_src_on_path()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from agent_knowledgebase.config import Settings  # noqa: E402
from agent_knowledgebase.models import SourceType  # noqa: E402
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService  # noqa: E402

SMALLEST_VID = "yJK5GueSHmU"


def main() -> None:
    kb_id = resolve_kb_id()
    print("=== Building service ===")
    t0 = time.monotonic()
    settings = Settings()
    print(f"  embedding_provider={settings.embedding_provider}")
    print(f"  embedding_model={settings.embedding_model}")
    print(f"  embed_base_url={getattr(settings, 'embed_base_url', None)!r}")
    print(f"  embed_timeout_seconds={settings.embed_timeout_seconds}")
    print(f"  kb_backend={settings.kb_backend}")
    print(f"  saves_dir={settings.saves_dir}")
    print(f"  build_settings_done {time.monotonic() - t0:.2f}s")

    t0 = time.monotonic()
    svc = KnowledgebaseService(settings)
    print(f"  build_service_done {time.monotonic() - t0:.2f}s")

    target = next(p for p in transcripts_dir().glob(f"*{SMALLEST_VID}*.md"))
    print(f"\n=== Target: {target.name} ({target.stat().st_size} bytes) ===")

    # Clear any prior failed source.
    srcs = svc.list_sources(kb_id)
    for s in srcs:
        if getattr(s, "dedup_key", None) == f"yt:{SMALLEST_VID}":
            print(f"  removing prior source {s.id} (status={s.status})")
            svc.remove_source(s.id)

    print("\n=== Calling ingest_source (force-add) ===")
    t0 = time.monotonic()
    try:
        result = svc.ingest_source(
            kb_id,
            SourceType.file,
            str(target),
            metadata={
                "video_id": SMALLEST_VID,
                "kb_source_label": f"yt-transcript:{SMALLEST_VID}",
            },
            dedup_key=f"yt:{SMALLEST_VID}",
            dedup_policy="force-add",
        )
        elapsed = time.monotonic() - t0
        print(f"\n=== DONE in {elapsed:.1f}s ===")
        print(f"  status={result.status}  chunks={result.chunk_count}  source_id={result.id}")
    except BaseException as exc:
        elapsed = time.monotonic() - t0
        print(f"\n=== FAILED after {elapsed:.1f}s ===")
        print(f"  {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
