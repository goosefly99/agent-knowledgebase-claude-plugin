#!/bin/sh
# Idle main process so the container stays running and `docker exec`
# can launch a fresh MCP server per session. Trap SIGTERM so
# `docker stop` shuts down cleanly.
set -eu
trap 'exit 0' TERM INT
# `tail -f` rather than `sleep infinity` so the PID 1 forwards stdio
# correctly under `docker logs`.
tail -f /dev/null &
wait $!
