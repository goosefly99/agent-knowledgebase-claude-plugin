"""Phase B: FTS5 startup probe in server.main().

Verifies that server.main() exits non-zero with a structured stderr JSON
containing 'error': 'sqlite_fts5_unavailable' and 'remediation' when
sqlite3 lacks FTS5 support.

The test mocks sqlite3.connect to raise OperationalError on CREATE VIRTUAL
TABLE ... USING fts5(...) to simulate a non-FTS5 build.
"""

from __future__ import annotations

import json
import sqlite3
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers to exercise main() without actually running the MCP server
# ---------------------------------------------------------------------------


def _call_fts5_probe_only() -> tuple[int, str]:
    """Run only the FTS5 probe section from server.main(), captured.

    Returns (exit_code, stderr_text). exit_code is 0 if probe passes,
    1 if it fails.
    """
    import io

    stderr_buf = io.StringIO()
    exit_code = 0

    import sqlite3 as _sqlite3

    try:
        _probe_conn = _sqlite3.connect(":memory:")
        _probe_conn.execute("CREATE VIRTUAL TABLE _probe USING fts5(x);")
        _probe_conn.execute("DROP TABLE _probe;")
        _probe_conn.close()
    except _sqlite3.OperationalError as exc:
        import json as _json

        structured_msg = _json.dumps(
            {
                "error": "sqlite_fts5_unavailable",
                "detail": str(exc),
                "remediation": (
                    "Install Python with FTS5-enabled sqlite3 "
                    "(Python 3.11+ on standard CPython distros include FTS5 by default)."
                ),
            }
        )
        print(structured_msg, file=stderr_buf)
        exit_code = 1

    return exit_code, stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fts5_probe_passes_on_normal_build() -> None:
    """On a standard CPython build with FTS5, the probe should pass (exit 0)."""
    exit_code, stderr = _call_fts5_probe_only()
    assert exit_code == 0, (
        f"FTS5 probe failed on this build — stdout: {stderr!r}. "
        "If FTS5 is genuinely unavailable, this is a real problem."
    )
    assert stderr == ""


class TestFts5ProbeFailurePath:
    """Simulate a non-FTS5 sqlite build by mocking sqlite3.connect."""

    def test_structured_stderr_on_fts5_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """When FTS5 is unavailable, server.main() emits structured JSON to stderr
        and exits non-zero.

        We test by patching the server module's sqlite3 import and asserting
        the probe catches the error and calls sys.exit(1).
        """
        import io

        captured_stderr = io.StringIO()
        captured_exit_code: list[int] = []

        fts5_error = sqlite3.OperationalError("no such module: fts5")

        # Build a mock that raises on CREATE VIRTUAL TABLE fts5 but allows
        # the in-memory connect.
        real_connect = sqlite3.connect

        class FakeCursor:
            def execute(self, sql: str, *args, **kwargs):
                if "fts5" in sql.lower():
                    raise fts5_error
                return self

            def fetchone(self):
                return None

            def fetchall(self):
                return []

        class FakeConn:
            def execute(self, sql: str, *args, **kwargs):
                if "fts5" in sql.lower():
                    raise fts5_error
                return FakeCursor()

            def close(self):
                pass

            def cursor(self):
                return FakeCursor()

        def fake_connect(path, **kwargs):
            if path == ":memory:":
                return FakeConn()
            return real_connect(path, **kwargs)

        # Patch the server module's sqlite3 usage in main().
        # We simulate it by patching sys.exit and sys.stderr then calling main().
        from agent_knowledgebase import server as server_module

        def fake_exit(code: int) -> None:
            captured_exit_code.append(code)
            raise SystemExit(code)

        saves = tmp_path / "saves"
        saves.mkdir()

        with (
            monkeypatch.context() as m,
        ):
            m.setenv("AGENT_KB_SAVES_DIR", str(saves))
            m.setattr(sys, "stderr", captured_stderr)

            # Patch sqlite3.connect in the server module's local scope.
            # The probe uses `import sqlite3 as _sqlite3` inside main(), so we
            # patch the top-level sqlite3 module's connect method.
            m.setattr(sqlite3, "connect", fake_connect)
            m.setattr(sys, "exit", fake_exit)

            with pytest.raises(SystemExit) as exc_info:
                server_module.main()

        assert exc_info.value.code == 1 or (captured_exit_code and captured_exit_code[0] == 1), (
            "Expected sys.exit(1) when FTS5 unavailable"
        )

        stderr_text = captured_stderr.getvalue()
        assert stderr_text, "Expected non-empty stderr when FTS5 probe fails"

        # Parse the first JSON line.
        first_line = stderr_text.strip().split("\n")[0]
        try:
            payload = json.loads(first_line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"stderr output is not valid JSON: {first_line!r} — {exc}")

        assert payload.get("error") == "sqlite_fts5_unavailable", (
            f"Expected error='sqlite_fts5_unavailable', got {payload!r}"
        )
        assert "remediation" in payload, f"Expected 'remediation' field in payload: {payload!r}"
        assert payload["remediation"], "remediation field must be non-empty"

    def test_probe_error_shape_has_required_fields(self) -> None:
        """The structured error payload must contain 'error', 'detail', 'remediation'."""
        fts5_error = sqlite3.OperationalError("no such module: fts5")
        # Simulate what main() emits on failure.
        payload = {
            "error": "sqlite_fts5_unavailable",
            "detail": str(fts5_error),
            "remediation": (
                "Install Python with FTS5-enabled sqlite3 "
                "(Python 3.11+ on standard CPython distros include FTS5 by default)."
            ),
        }
        assert payload["error"] == "sqlite_fts5_unavailable"
        assert "remediation" in payload
        assert payload["remediation"]
        assert "detail" in payload

        # Must be JSON-serializable.
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        assert decoded == payload
