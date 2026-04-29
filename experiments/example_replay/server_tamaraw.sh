#!/usr/bin/env bash
# Start server.py with the tamaraw preset.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/_common.sh"

exec python "${DIR}/server.py" \
    --dst_port "${DST_PORT}" \
    --preset tamaraw "$@"
