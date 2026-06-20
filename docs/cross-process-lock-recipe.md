# Cross-process lock recipes for `kb_ingest_batch`

The kb server serializes concurrent ingest against the same `kb_id` by
holding an in-process `threading.Lock` for the entire pipeline duration
(see `src/agent_knowledgebase/services/knowledgebase.py`). That lock
only coordinates callers inside a single Python process. If a second
Python process (a second `uv run`, a second MCP server instance, a
Celery worker, etc.) ingests into the same `<saves_dir>/<kb>/`
directory, the two processes can race — ChromaDB writes, SQLite
metadata writes, and the `kb_pipeline_status` rows all drift out of
sync. This document lists four opt-in recipes for cross-process
coordination. None of them is a runtime dependency of this plugin.
Pick whichever fits your deployment and wrap the MCP `kb_ingest_batch`
call on your side.

All recipes assume a stable lock path derived from the target `kb_id`,
for example `<saves_dir>/<sanitized-kb-name>/.ingest.lock`.

---

## Recipe 1 — `filelock` (primary, recommended)

`filelock` is the most widely used cross-platform file-lock library on
PyPI. It is pure-Python, works on Windows and POSIX, has a clean
context-manager API, and handles timeout / stale-lock semantics
predictably. This is the recipe we recommend for most multi-process
deployments.

```bash
pip install filelock
```

Note: `filelock` is **not** added to `pyproject.toml`. Install it
yourself in the environment that wraps `kb_ingest_batch`.

### Sync wrapper

```python
from pathlib import Path

from filelock import FileLock, Timeout


def ingest_with_filelock(kb_saves_dir: Path, kb_name: str, *, call_kb_ingest_batch):
    """Wrap ``kb_ingest_batch`` with a cross-process file lock."""
    lock_path = kb_saves_dir / kb_name / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=300)  # 5 minutes
    try:
        with lock:
            return call_kb_ingest_batch()
    except Timeout:
        raise RuntimeError(
            f"timed out acquiring cross-process ingest lock on {lock_path}"
        )
```

### Asyncio-friendly wrapper

`filelock` itself is sync. When the caller lives in an asyncio loop,
offload the blocking acquire/release to a worker thread so the loop
keeps serving other work while we wait:

```python
import asyncio
from pathlib import Path

from filelock import FileLock, Timeout


async def ingest_with_filelock_async(
    kb_saves_dir: Path,
    kb_name: str,
    *,
    call_kb_ingest_batch,
):
    lock_path = kb_saves_dir / kb_name / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=300)

    await asyncio.to_thread(lock.acquire)
    try:
        # ``call_kb_ingest_batch`` may itself be sync — wrap it in
        # ``asyncio.to_thread`` too if so.
        return await asyncio.to_thread(call_kb_ingest_batch)
    finally:
        lock.release()
```

Why `filelock` first: cleanest API, cross-platform, the most popular
option on PyPI, and its context-manager form composes naturally with
the rest of your pipeline.

---

## Recipe 2 — `portalocker` (primary alternative)

`portalocker` is a well-maintained alternative that is especially
pleasant on Windows. It exposes advisory and exclusive modes, handles
crashed-process cleanup reliably (the lock releases when the holder's
file descriptor is dropped by the OS), and has a long track record in
Windows-heavy deployments.

```bash
pip install portalocker
```

### Sync wrapper

```python
from pathlib import Path

import portalocker


def ingest_with_portalocker(kb_saves_dir: Path, kb_name: str, *, call_kb_ingest_batch):
    lock_path = kb_saves_dir / kb_name / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # ``EXCLUSIVE`` is a writer-lock; blocks until acquired.  Set a
    # timeout if you prefer a bounded wait.
    with portalocker.Lock(
        str(lock_path),
        mode="a",
        flags=portalocker.LOCK_EX,
        timeout=300,
    ):
        return call_kb_ingest_batch()
```

### Asyncio-friendly wrapper

```python
import asyncio
from pathlib import Path

import portalocker


async def ingest_with_portalocker_async(
    kb_saves_dir: Path,
    kb_name: str,
    *,
    call_kb_ingest_batch,
):
    lock_path = kb_saves_dir / kb_name / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _acquire_and_run():
        with portalocker.Lock(
            str(lock_path),
            mode="a",
            flags=portalocker.LOCK_EX,
            timeout=300,
        ):
            return call_kb_ingest_batch()

    return await asyncio.to_thread(_acquire_and_run)
```

Why choose `portalocker` over `filelock`: preferred on Windows-only
fleets, and the OS-level cleanup on process death is more predictable
in practice. If your deployment is POSIX-only, `filelock` is the
usual pick.

---

## Recipe 3 — Postgres advisory lock (DB-backed alternative)

If your deployment already runs Postgres and you have migrated the
kb's metadata layer to Postgres (or share a Postgres instance for
orchestration), Postgres advisory locks replace the file-based lock
entirely. Advisory locks are fast, scoped to the transaction or
session, and automatically released when the transaction commits or
rolls back — including on crash.

The pattern is: hash the `kb_id` to a 32-bit integer, take an
`xact_lock` inside a transaction, run the ingest, let the transaction
commit. Two concurrent callers with the same `kb_id` will serialize
on the lock.

```python
from sqlalchemy import create_engine, text


def ingest_with_pg_advisory(dsn: str, kb_id: str, *, call_kb_ingest_batch):
    """Serialize on a Postgres advisory lock keyed by ``kb_id``."""
    engine = create_engine(dsn, pool_pre_ping=True)
    with engine.begin() as conn:
        # ``hashtext`` folds the kb_id into a 32-bit int; ``xact`` scope
        # releases the lock automatically when the transaction ends.
        conn.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:kb_id))"),
            {"kb_id": kb_id},
        )
        return call_kb_ingest_batch()
```

For asyncio, use `asyncpg` directly or `SQLAlchemy` with `asyncio.to_thread`
around the sync call as in Recipe 1. Note: if your kb's own metadata
DB is also Postgres, prefer a shared session so the lock lives on the
same connection pool.

Why Postgres: if you already operate it, you skip the file-system
dependency entirely — no lock file, no stale-FD edge cases, no
NFS-flock caveats. The cost is a runtime Postgres requirement.

---

## Recipe 4 — Raw `fcntl.flock` / `msvcrt.locking` (no-deps baseline)

When you cannot add a dependency and can accept sharper edges, the
standard library gives you `fcntl.flock` on POSIX and
`msvcrt.locking` on Windows. There is no built-in timeout helper, you
are responsible for the file-descriptor lifecycle, and the
error-handling shapes differ across platforms.

Dispatch on `sys.platform`:

```python
import os
import sys
from pathlib import Path


def ingest_with_raw_os_lock(kb_saves_dir: Path, kb_name: str, *, call_kb_ingest_batch):
    lock_path = kb_saves_dir / kb_name / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if sys.platform.startswith("win"):
            import msvcrt

            # LK_LOCK blocks for up to ~10s per call; retry to extend.
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                return call_kb_ingest_batch()
            finally:
                # Rewind before unlocking the same byte range.
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                return call_kb_ingest_batch()
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
```

For asyncio, wrap the whole call in `asyncio.to_thread` — the same
pattern as Recipe 1's async form.

Why raw OS locks: zero runtime dependencies. The trade-off is you own
the FD lifecycle, the timeout behaviour differs per platform, and
there is no graceful-timeout API. Unless you specifically cannot add
a dep, prefer Recipe 1 or 2.

---

## Observability aid

The `kb_pipeline_status` row (v2 in v0.6.0, augmented with
`request_id` and `tool_caller_version`) makes concurrent multi-process
runs visible after the fact. If two processes somehow bypass the
in-process lock — for example because you have not wrapped one of
them in a recipe above — each still writes its own `pipeline_runs`
row. Correlate by wall-clock overlap between `started_at` /
`ended_at` and compare `request_id` / `tool_caller_version`; a
collision is a clear signal that cross-process coordination is
missing. The `knowledgebase_stderr_log` emissions (`phase="lock"`,
`lock_acquire` / `lock_release`, milliseconds-held) give a matching
per-line trace.

This is an after-the-fact drift detector, not a prevention tool. Use
it alongside one of the recipes above, not in place of them.

---

## Forward pointer

Enforcement of cross-process locking is deferred. The v0.4.0 planning
item tracks graduating ONE of the recipes above into a runtime
dependency (most likely `filelock` given the trade-offs) once the
exit criterion is met. Until then, all four recipes are opt-in and
the built-in `threading.Lock` remains single-process-only. See
`ROADMAP.md` for the release-gate checkbox tracking this item.
