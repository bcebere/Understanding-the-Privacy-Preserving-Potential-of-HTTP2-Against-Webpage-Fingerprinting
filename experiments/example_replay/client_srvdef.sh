#!/usr/bin/env bash
# Run client.py with server defenses.
# Set CAPTURE=0 to skip pcap recording (no sudo needed).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/_common.sh"

CMD=(python "${DIR}/client.py"
     --dst_ip   "${DST_IP}"
     --dst_port "${DST_PORT}"
     --defense  nop
     --repeats  "${REPEATS}"
     --subpage_limit "${SUBPAGE_LIMIT}"
     --capture  "${CAPTURE}"
     --request_server_defense "${REQUEST_SERVER_DEFENSE}")

if [[ "${CAPTURE}" == "1" ]]; then
    CMD+=(--ifname "${IFNAME}")
fi

exec "${CMD[@]}" "$@"
