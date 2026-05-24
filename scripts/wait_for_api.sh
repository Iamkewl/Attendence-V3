#!/usr/bin/env bash
# wait_for_api.sh — poll http://localhost:8000/healthz until HTTP 200 or 60s timeout.
set -euo pipefail

URL="http://localhost:8000/healthz"
TIMEOUT=60
INTERVAL=2
elapsed=0

echo "Waiting for API at ${URL} ..."

while true; do
    if curl -fsS --max-time 3 "${URL}" >/dev/null 2>&1; then
        echo "API is ready (${elapsed}s elapsed)."
        exit 0
    fi

    if [ "${elapsed}" -ge "${TIMEOUT}" ]; then
        echo "ERROR: API did not become ready within ${TIMEOUT}s." >&2
        exit 1
    fi

    sleep "${INTERVAL}"
    elapsed=$(( elapsed + INTERVAL ))
    echo "  still waiting... (${elapsed}s)"
done
