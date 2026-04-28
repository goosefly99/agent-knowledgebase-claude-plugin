"""Probe the agent-knowledgebase MCP server's `tools/list` response over stdio.

Sanity check: confirms the server starts cleanly, advertises tool capability,
and exposes a non-empty tool list. Useful for diagnosing handshake/tool
registration regressions before running heavier smoke scripts.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from _helpers import plugin_root


def main() -> None:
    root = plugin_root()
    server = root / "bin" / "run_server.py"
    py = (root / ".venv" / "Scripts" / "python.exe") if sys.platform == "win32" else (
        root / ".venv" / "bin" / "python"
    )
    if not py.exists():
        py = Path(sys.executable)

    env = {**os.environ, "CLAUDE_PLUGIN_ROOT": str(root)}

    proc = subprocess.Popen(
        [str(py), str(server)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(root),
    )

    def send(msg: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def recv() -> dict | None:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        return json.loads(line) if line else None

    send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "probe", "version": "0.0.1"},
    }})
    init_resp = recv()
    print("INIT_RESP_KEYS:", list(init_resp.keys()) if init_resp else None)
    if init_resp and "result" in init_resp:
        caps = init_resp["result"].get("capabilities", {})
        print("CAPABILITIES_KEYS:", list(caps.keys()))
        print("INIT_HAS_TOOLS_CAP:", "tools" in caps)
        sinfo = init_resp["result"].get("serverInfo", {})
        print("SERVER_INFO:", json.dumps(sinfo))
        print("PROTOCOL_VERSION_RESP:", init_resp["result"].get("protocolVersion"))

    send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools_resp = recv()
    print("TOOLS_RESP_KEYS:", list(tools_resp.keys()) if tools_resp else None)
    if tools_resp and "result" in tools_resp:
        tools = tools_resp["result"].get("tools", [])
        print("TOOL_COUNT:", len(tools))
        print("FIRST_5:", [t.get("name") for t in tools[:5]])
    elif tools_resp and "error" in tools_resp:
        print("TOOLS_ERR:", json.dumps(tools_resp["error"])[:400])

    stderr_data = ""
    try:
        proc.terminate()
        _, stderr_data = proc.communicate(timeout=5)
    except Exception:
        proc.kill()
    print("STDERR_TAIL:")
    print(stderr_data[-2000:] if stderr_data else "<empty>")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
