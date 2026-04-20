"""Tests for :func:`knowledgebase_stderr_log` — structured stderr emission."""

from __future__ import annotations

import json

import pytest

from agent_knowledgebase.services.stderr_log import knowledgebase_stderr_log


# Expected required keys on every emission.
_REQUIRED_FIELDS = [
    "kb_id",
    "op",
    "phase",
    "elapsed_ms",
    "rows_in",
    "rows_ok",
    "rows_skipped",
    "rows_failed",
    "dedup_policy",
    "request_id",
    "tool_caller_version",
]


def test_knowledgebase_stderr_log_emits_structured_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Full payload round-trips through ``json.loads`` with all documented fields present."""
    knowledgebase_stderr_log(
        kb_id="kb-abc",
        op="ingest_batch_end",
        phase="post_run",
        elapsed_ms=123,
        rows_in=5,
        rows_ok=4,
        rows_skipped=1,
        rows_failed=0,
        dedup_policy="skip",
        request_id="req-xyz",
        tool_caller_version="1.2.3",
    )
    captured = capsys.readouterr()
    # Exactly one line, ending in newline.
    assert captured.err.endswith("\n")
    line = captured.err.strip()
    payload = json.loads(line)

    # All 11 required fields present.
    for key in _REQUIRED_FIELDS:
        assert key in payload, f"missing field {key!r} in {payload!r}"

    assert payload["kb_id"] == "kb-abc"
    assert payload["op"] == "ingest_batch_end"
    assert payload["phase"] == "post_run"
    assert payload["elapsed_ms"] == 123
    assert payload["rows_in"] == 5
    assert payload["rows_ok"] == 4
    assert payload["rows_skipped"] == 1
    assert payload["rows_failed"] == 0
    assert payload["dedup_policy"] == "skip"
    assert payload["request_id"] == "req-xyz"
    assert payload["tool_caller_version"] == "1.2.3"


def test_knowledgebase_stderr_log_error_code_absent_on_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Success emissions omit ``error_code``/``error_message`` entirely."""
    knowledgebase_stderr_log(
        kb_id="kb-1",
        op="ingest_source",
        phase="finalize",
        elapsed_ms=50,
        rows_in=1,
        rows_ok=1,
        rows_skipped=0,
        rows_failed=0,
        dedup_policy="skip",
        request_id=None,
        tool_caller_version=None,
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip())
    assert "error_code" not in payload
    assert "error_message" not in payload


def test_knowledgebase_stderr_log_error_code_present_on_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failure emissions carry SCREAMING_SNAKE_CASE ``error_code``."""
    knowledgebase_stderr_log(
        kb_id="kb-2",
        op="lock_acquire",
        phase="lock",
        elapsed_ms=0,
        rows_in=0,
        rows_ok=0,
        rows_skipped=0,
        rows_failed=0,
        dedup_policy="n/a",
        request_id=None,
        tool_caller_version=None,
        error_code="LOCK_ACQUIRE_FAILED",
        error_message="timeout",
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip())
    assert payload["error_code"] == "LOCK_ACQUIRE_FAILED"
    # SCREAMING_SNAKE_CASE contract: all upper + underscores only.
    assert payload["error_code"] == payload["error_code"].upper()
    assert " " not in payload["error_code"]
    assert "-" not in payload["error_code"]
    assert payload["error_message"] == "timeout"


def test_knowledgebase_stderr_log_exactly_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exactly one newline-terminated line per call."""
    knowledgebase_stderr_log(
        kb_id="kb-3",
        op="lock_release",
        phase="lock",
        elapsed_ms=17,
        rows_in=0,
        rows_ok=0,
        rows_skipped=0,
        rows_failed=0,
        dedup_policy="n/a",
        request_id=None,
        tool_caller_version=None,
    )
    captured = capsys.readouterr()
    # Exactly one line-break character produced.
    assert captured.err.count("\n") == 1
    # The content before the newline must be valid JSON.
    line = captured.err[:-1]
    assert "\n" not in line
    json.loads(line)  # does not raise
