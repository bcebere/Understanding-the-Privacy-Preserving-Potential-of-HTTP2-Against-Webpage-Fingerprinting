#!/usr/bin/env bash
# Run client.py with the HTTPOS defense.
# Set CAPTURE=0 to skip pcap recording (no sudo needed).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/_common.sh"

CMD=(python "${DIR}/client.py"
     --dst_ip   "${DST_IP}"
     --dst_port "${DST_PORT}"
     --defense  httpos
     --repeats  "${REPEATS}"
     --subpage_limit "${SUBPAGE_LIMIT}"
     --capture  "${CAPTURE}")

if [[ "${CAPTURE}" == "1" ]]; then
    CMD+=(--ifname "${IFNAME}")
fi
if [[ -n "${REQUEST_SERVER_DEFENSE}" ]]; then
    CMD+=(--request_server_defense "${REQUEST_SERVER_DEFENSE}")
fi

exec "${CMD[@]}" "$@"
