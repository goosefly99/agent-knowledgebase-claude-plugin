"""Persistent JSON-RPC stdio client for the agent-knowledgebase MCP server.

Used by the integration smoke scripts to drive the real server in a
subprocess. Spawns `<plugin_root>/bin/run_server.py` under the project's
`.venv` python, drains stderr in a background thread, and exposes a
synchronous `call(tool_name, **arguments)` helper that returns the parsed
JSON of the first text-content block (or the raw text if not JSON).

Usage:
    from mcp_client import KbClient
    with KbClient() as kb:
        print(kb.call("kb_list"))
        kb.call("kb_create", name="claude-rag", description="agentic RAG KB")
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "bin" / "run_server.py"


def _resolve_python() -> Path:
    """Return the python interpreter to launch the server with.

    Prefer the project venv (`<root>/.venv/Scripts/python.exe` on Windows or
    `<root>/.venv/bin/python` elsewhere); fall back to the running
    interpreter so the client still works in CI or other environments where
    the venv layout differs.
    """
    if sys.platform == "win32":
        candidate = ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


class KbClient:
    """Spawn the MCP server, perform the initialize handshake, and dispatch tool calls."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self._id = 0
        self._stderr_buf: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> "KbClient":
        env = {**os.environ, "CLAUDE_PLUGIN_ROOT": str(ROOT)}
        self.proc = subprocess.Popen(
            [str(_resolve_python()), str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(ROOT),
        )

        # Drain stderr in a background thread so it never blocks stdout.
        def _drain() -> None:
            assert self.proc is not None and self.proc.stderr is not None
            for line in self.proc.stderr:
                self._stderr_buf.append(line)

        self._stderr_thread = threading.Thread(target=_drain, daemon=True)
        self._stderr_thread.start()

        self._send({
            "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "agent-kb-integration", "version": "0.0.1"},
            },
        })
        init = self._recv()
        if init is None or "result" not in init:
            raise RuntimeError(f"initialize failed: {init}")
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    # ---- low level ----
    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, msg: dict) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _recv(self, timeout: float = 600.0) -> dict | None:
        assert self.proc is not None and self.proc.stdout is not None
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"server exited rc={self.proc.returncode}; "
                        f"stderr_tail={''.join(self._stderr_buf)[-2000:]}"
                    )
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON line on stdout (rare); skip.
                continue
        raise TimeoutError(f"no response within {timeout}s")

    # ---- high level ----
    def call(self, tool_name: str, **arguments) -> object:
        """Call a tool. Return parsed JSON of its first text-content block, or raw text."""
        rid = self._next_id()
        self._send({
            "jsonrpc": "2.0", "id": rid, "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        resp = self._recv()
        if resp is None:
            raise RuntimeError("no response")
        if "error" in resp:
            raise RuntimeError(f"{tool_name} -> error: {resp['error']}")
        result = resp.get("result", {})
        if result.get("isError"):
            raise RuntimeError(f"{tool_name} -> isError: {result}")
        contents = result.get("content", [])
        if not contents:
            return None
        text = contents[0].get("text", "")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    def stderr_tail(self, n: int = 2000) -> str:
        return "".join(self._stderr_buf)[-n:]
