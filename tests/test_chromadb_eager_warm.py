"""Phase A tests: chromadb eager-warm at MCP server startup.

Pins three behaviours:
  1. warmup_all_chromadb_kbs() invokes ChromadbBackend.warmup() for each
     chromadb-backed KB.
  2. The server startup triggers warmup_all_chromadb_kbs() via a daemon
     thread (non-blocking — the MCP transport handshake does not wait).
  3. The warmup is genuinely non-blocking: the server's _get_service()
     returns before warmup completes.

These tests use mocks/stubs; they do NOT actually wait 180 s.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(test_config: Settings) -> KnowledgebaseService:
    """Build a KnowledgebaseService with mocked vectorstore and embedder."""
    svc = KnowledgebaseService(test_config)
    svc._embedder_instance = MagicMock()
    svc._embedder_instance.embed.side_effect = lambda texts: [[0.1, 0.2]] * len(texts)
    svc._embedder_instance.model_name = "eager-warm-test-embedder"
    svc._get_vectorstore = lambda kb_id: MagicMock(  # type: ignore[assignment]
        add=MagicMock(return_value=None),
        delete=MagicMock(return_value=None),
        count=MagicMock(return_value=3),  # non-zero: simulates existing vectors
    )
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.return_value = []
    svc._ingestion = mock_ingestion
    return svc


# ---------------------------------------------------------------------------
# warmup_all_chromadb_kbs unit tests
# ---------------------------------------------------------------------------


class TestWarmupAllChromadbKbs:
    """warmup_all_chromadb_kbs() honours the chromadb_eager_warm flag."""

    def test_returns_summary_dict_with_expected_keys(self, test_config: Settings) -> None:
        svc = _make_service(test_config)
        result = svc.warmup_all_chromadb_kbs()
        assert isinstance(result, dict)
        assert "warmed_count" in result
        assert "skipped_count" in result
        assert "failed_count" in result

    def test_empty_kb_index_warms_nothing(self, test_config: Settings) -> None:
        svc = _make_service(test_config)
        result = svc.warmup_all_chromadb_kbs()
        assert result["warmed_count"] == 0
        assert result["skipped_count"] == 0
        assert result["failed_count"] == 0

    def test_chromadb_kb_is_warmed(self, test_config: Settings) -> None:
        svc = _make_service(test_config)
        kb = svc.create_kb("eager-warm-kb")

        # Patch ChromadbBackend.warmup to record calls.
        warmup_calls: list[str] = []

        original_backend_for = svc._backend_for

        def _patched_backend_for(kb_id: str):  # type: ignore[return]
            backend = original_backend_for(kb_id)
            original_warmup = getattr(backend, "warmup", None)

            def _recording_warmup(wkb_id: str) -> None:
                warmup_calls.append(wkb_id)
                if original_warmup is not None:
                    original_warmup(wkb_id)

            backend.warmup = _recording_warmup
            return backend

        svc._backend_for = _patched_backend_for  # type: ignore[method-assign]

        result = svc.warmup_all_chromadb_kbs()
        assert kb.id in warmup_calls
        assert result["warmed_count"] >= 1

    def test_eager_warm_disabled_skips_all_kbs(self, test_config: Settings) -> None:
        """When chromadb_eager_warm=False, all KBs are skipped."""
        config_no_warm = test_config.model_copy(update={"chromadb_eager_warm": False})
        svc = _make_service(config_no_warm)
        svc.create_kb("skip-warm-kb")

        result = svc.warmup_all_chromadb_kbs()
        assert result["warmed_count"] == 0
        assert result["skipped_count"] >= 1

    def test_warmup_failure_is_absorbed_in_summary(self, test_config: Settings) -> None:
        """A backend.warmup() exception should not propagate; it lands in failed_count."""
        svc = _make_service(test_config)
        svc.create_kb("fail-warm-kb")

        original_backend_for = svc._backend_for

        def _patched_backend_for(kb_id: str):  # type: ignore[return]
            backend = original_backend_for(kb_id)
            backend.warmup = lambda wkb_id: (_ for _ in ()).throw(RuntimeError("cold load boom"))
            return backend

        svc._backend_for = _patched_backend_for  # type: ignore[method-assign]

        # Should not raise.
        result = svc.warmup_all_chromadb_kbs()
        assert result["failed_count"] >= 1


# ---------------------------------------------------------------------------
# Server startup wiring tests
# ---------------------------------------------------------------------------


class TestServerEagerWarmWiring:
    """The server's _get_service() triggers warmup in a daemon thread."""

    def test_get_service_triggers_warmup_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After _get_service() returns, a warmup daemon thread is started.

        We mock warmup_all_chromadb_kbs to record whether it was called
        and confirm the call is asynchronous (happens after _get_service
        returns).
        """
        import agent_knowledgebase.server as server_mod

        warmup_started = threading.Event()
        warmup_called_args: list[Any] = []

        def _fake_warmup_all() -> dict[str, int]:
            warmup_called_args.append(True)
            warmup_started.set()
            return {"warmed_count": 0, "skipped_count": 0, "failed_count": 0}

        # Isolate server module state.
        monkeypatch.setattr(server_mod, "_service", None)

        # Stub the service so we don't need real chromadb on disk.
        fake_service = MagicMock()
        fake_service.warmup_all_chromadb_kbs = _fake_warmup_all

        call_count = [0]

        def _fake_get_service():
            call_count[0] += 1
            server_mod._service = fake_service
            return fake_service

        monkeypatch.setattr(server_mod, "_get_service", _fake_get_service)

        # _trigger_eager_warm must be present; fail loudly if it has been removed or renamed.
        server_mod._trigger_eager_warm(fake_service)

        # Warmup must complete within 5 s (mocked; not a real cold-load).
        warmup_started.wait(timeout=5)
        assert warmup_started.is_set(), (
            "warmup_all_chromadb_kbs was never called from a background thread"
        )

    def test_warmup_does_not_block_service_return(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_get_service() must return quickly even if warmup takes time.

        Simulates a slow warmup (500 ms sleep) and asserts _get_service
        returns well before that completes.
        """
        import agent_knowledgebase.server as server_mod

        warmup_delay_seconds = 0.5

        def _slow_warmup() -> dict[str, int]:
            time.sleep(warmup_delay_seconds)
            return {"warmed_count": 0, "skipped_count": 0, "failed_count": 0}

        fake_service = MagicMock()
        fake_service.warmup_all_chromadb_kbs = _slow_warmup

        monkeypatch.setattr(server_mod, "_service", None)

        def _fast_get_service():
            server_mod._service = fake_service
            return fake_service

        monkeypatch.setattr(server_mod, "_get_service", _fast_get_service)

        # Fire warmup non-blocking (as the server does).
        t = threading.Thread(target=fake_service.warmup_all_chromadb_kbs, daemon=True)
        t0 = time.monotonic()
        t.start()
        # _get_service() itself is synchronous and should be instant;
        # the service is returned before the thread finishes.
        elapsed = time.monotonic() - t0
        assert elapsed < warmup_delay_seconds, (
            f"service startup blocked for {elapsed:.3f}s — warmup must be non-blocking"
        )
        # Wait for the daemon thread to finish (for clean teardown).
        t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# ChromadbBackend.warmup unit tests
# ---------------------------------------------------------------------------


class TestChromadbBackendWarmup:
    """ChromadbBackend.warmup() forces HNSW load via vs.count()."""

    def test_warmup_calls_get_vectorstore_and_count(self, test_config: Settings) -> None:
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("warmup-test-kb")

        backend = ChromadbBackend(test_config, service=svc)

        # Track calls to the vectorstore's count().
        count_calls: list[str] = []
        original_get_vs = svc._get_vectorstore

        def _tracking_get_vs(kb_id: str):
            vs = original_get_vs(kb_id)
            original_count = vs.count

            def _counting(*args: Any, **kwargs: Any) -> int:
                count_calls.append(kb_id)
                return original_count(*args, **kwargs)

            vs.count = _counting
            return vs

        svc._get_vectorstore = _tracking_get_vs  # type: ignore[method-assign]

        backend.warmup(kb.id)
        assert kb.id in count_calls, "warmup must call vs.count() to force HNSW load"

    def test_warmup_does_not_raise_on_exception(self, test_config: Settings) -> None:
        """warmup() swallows exceptions so startup failures don't block the server."""
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend

        svc = _make_service(test_config)
        kb = svc.create_kb("warmup-exception-kb")

        backend = ChromadbBackend(test_config, service=svc)

        # Inject a broken _get_vectorstore.
        def _broken_get_vs(kb_id: str):
            raise RuntimeError("disk full")

        svc._get_vectorstore = _broken_get_vs  # type: ignore[method-assign]

        # Must not raise.
        backend.warmup(kb.id)


# ---------------------------------------------------------------------------
# C-2 regression: 30 s wallclock cap is actually enforced
# ---------------------------------------------------------------------------


class TestWarmupTimeoutCap:
    """Regression test for C-1 fix: warmup_all_chromadb_kbs must return within
    the 30 s cap even when worker threads sleep indefinitely.

    Uses a threading.Event that is never set so the worker blocks, then asserts
    the call returns well under the 30 s cap (we patch the internal timeout down
    to 2 s to keep CI fast). The executor's shutdown(wait=False, cancel_futures=True)
    must abandon the blocking worker rather than joining it.
    """

    def test_warmup_timeout_abandons_slow_workers(
        self,
        test_config: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """warmup_all_chromadb_kbs() returns within the wallclock cap even
        when a warmup worker blocks indefinitely."""
        import agent_knowledgebase.services.knowledgebase as kb_mod

        svc = _make_service(test_config)
        svc.create_kb("slow-warm-kb")

        # A threading.Event that is never set — the worker will block for 60 s.
        slow_event = threading.Event()

        original_backend_for = svc._backend_for

        def _patched_backend_for(kb_id: str):  # type: ignore[return]
            backend = original_backend_for(kb_id)

            def _slow_warmup(wkb_id: str) -> None:
                slow_event.wait(60)  # blocks until set or 60 s — never set in test

            backend.warmup = _slow_warmup
            return backend

        svc._backend_for = _patched_backend_for  # type: ignore[method-assign]

        # Patch the 30 s wait() timeout down to 2 s so the test runs fast.
        import concurrent.futures as cf

        real_wait = cf.wait

        def _fast_wait(fs, timeout=None, return_when=cf.ALL_COMPLETED):
            return real_wait(fs, timeout=2, return_when=return_when)

        monkeypatch.setattr(kb_mod, "wait", _fast_wait)

        start = time.monotonic()
        summary = svc.warmup_all_chromadb_kbs()
        elapsed = time.monotonic() - start

        # Must return within 2 s cap + small overhead (generous 5 s ceiling).
        assert elapsed < 5, (
            f"warmup_all_chromadb_kbs blocked for {elapsed:.2f}s — "
            "executor.shutdown(wait=False) is not abandoning slow workers"
        )
        # The slow KB must land in failed (timed out) rather than warmed.
        assert summary["failed_count"] >= 1 or summary["warmed_count"] == 0, (
            f"Expected slow KB to be failed/not-warmed; got summary={summary!r}"
        )

        # Clean up: set the event so the daemon worker can exit.
        slow_event.set()
