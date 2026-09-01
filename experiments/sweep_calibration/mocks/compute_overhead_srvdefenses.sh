#!/usr/bin/env bash
# Overhead for one server-defense cell.  Client stays undefended.
#   ./compute_overhead_srvdefenses.sh <ip> <port> <defense> <level> <target> [pages]
#
# <target> is the domain that deploys the defense, or "all".
#   h2ps            -> the 1st-party domain (its claim is 1st-party-only)
#   alpaca/tamaraw  -> "all", or the leakiest server per Table 5
# Start the server elsewhere first: ./run_server_level.sh <port> <defense> <level>
set -u

IP="$1"
PORT="$2"
DEF="$3"
LVL="$4"
TARGET="$5"
PAGES="${6:-25}"

SCENARIO="`basename $PWD`"
WORKSPACE="/http2/experiments/sweep_calibration/${SCENARIO}/overhead"
export WF_DATASET="${WF_DATASET:-$SCENARIO}"
mkdir -p "$WORKSPACE"

# "all" stays unsuffixed; a domain becomes a short readable tag
if [ "$TARGET" = "all" ]; then
  TAG="srv${DEF}_${LVL}"
else
  TAG="srv${DEF}1p_${LVL}"
fi

if [ -f "$WORKSPACE/ovh_${TAG}.csv" ]; then
  echo "skip  $TAG"
  exit 0
fi

echo "=== $TAG  (target: $TARGET)"
H2_VERBOSE=0 python ./approximate_overhead.py \
    --dst_ip "$IP" --dst_port "$PORT" --pages "$PAGES" \
    --workspace "$WORKSPACE" --request_server_defense "$TARGET" --tag "$TAG"
