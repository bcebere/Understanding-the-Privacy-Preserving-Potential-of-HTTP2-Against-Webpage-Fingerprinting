#!/usr/bin/env bash
set -u

IP="$1"
PORT="$2"
IFACE="$3"
TAG="$4"
REPEATS="${5:-100}"

export WF_DATASET="$(basename "$PWD")"

echo "=== $TAG  $(date '+%F %T')"
python ./collect_traces.py --dst_ip "$IP" --dst_port "$PORT" --ifname "$IFACE" \
    --defense nop --request_server_defense all \
    --tag "$TAG" --repeats "$REPEATS"
