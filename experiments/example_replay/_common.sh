#!/usr/bin/env bash
# Override any of these by exporting before invoking a wrapper, e.g.:
#   DST_PORT=9443 ./client_front.sh
set -euo pipefail

DST_IP="${DST_IP:-127.0.0.1}"
DST_PORT="${DST_PORT:-8443}"
IFNAME="${IFNAME:-lo}"
REPEATS="${REPEATS:-50}"
SUBPAGE_LIMIT="${SUBPAGE_LIMIT:-20}"
CAPTURE="${CAPTURE:-1}"        # 0 to skip pcap
REQUEST_SERVER_DEFENSE="${REQUEST_SERVER_DEFENSE:-}"  # empty = none, "all" = every conn
