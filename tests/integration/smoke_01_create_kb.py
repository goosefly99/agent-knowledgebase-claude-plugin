"""Step 1: list existing KBs, create the test KB by name, verify.

Idempotent — if the KB (default name `claude-rag`) already exists, skip
creation and just print its info.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import require_env  # noqa: E402
from mcp_client import KbClient  # noqa: E402


def main() -> None:
    require_env(
        "AGENT_KB_SAVES_DIR",
        "set it to your persistent KB saves directory before running.",
    )
    name = os.environ.get("AGENT_KB_TEST_KB_NAME", "claude-rag")
    description = os.environ.get(
        "AGENT_KB_TEST_KB_DESCRIPTION",
        "Agentic RAG / knowledgebase methods — ingested from local YouTube transcripts.",
    )

    with KbClient() as kb:
        print("=== kb_list (before) ===")
        existing = kb.call("kb_list")
        print(existing)

        names = {row.get("name"): row.get("id") for row in (existing or [])}
        if name in names:
            kb_id = names[name]
            print(f"\n*** {name!r} already exists, id={kb_id}; skipping create ***")
        else:
            print(f"\n=== kb_create {name} ===")
            created = kb.call("kb_create", name=name, description=description)
            print(created)
            kb_id = created.get("id") if isinstance(created, dict) else None

        print("\n=== kb_info ===")
        info = kb.call("kb_info", kb_id=kb_id)
        print(info)

        print("\n=== KB_ID ===")
        print(kb_id)


if __name__ == "__main__":
    main()
