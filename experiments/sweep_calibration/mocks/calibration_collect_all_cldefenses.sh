#!/usr/bin/env bash
# PCAP collection for the client defenses.
#   ./collect_cldefenses.sh <dst_ip> <dst_port> <ifname> [repeats]
# Server must be running. Resumable: collect_traces.py skips existing captures.
#
#   LEVELS_front="vlow low lomid mid1" ./collect_cldefenses.sh ...
set -u

IP="$1"; PORT="$2"; IFACE="$3"; REPEATS="${4:-100}"
export WF_DATASET="$(basename "$PWD")"

DEFENSES=(front tamaraw h2pc httpos llama)

LEVELS_front="${LEVELS_front:-vlow low lomid mid1 mid2}" # high
LEVELS_tamaraw="${LEVELS_tamaraw:-vlow low lomid mid1 mid2}" # high
LEVELS_h2pc="${LEVELS_h2pc:-vlow low lomid mid1 mid2 high}"
LEVELS_httpos="${LEVELS_httpos:-vlow low lomid mid1 mid2}"
LEVELS_llama="${LEVELS_llama:-vlow low lomid mid1 mid2 high}"

run () {
  echo "=== $1 $2  $(date '+%F %T')"
  python ./collect_traces.py --dst_ip "$IP" --dst_port "$PORT" --ifname "$IFACE" \
      --defense "$1" --level "$2" --repeats "$REPEATS"
  echo "=== $1 $2 done  $(date '+%F %T')"
}

echo "dataset $WF_DATASET   repeats: $REPEATS"
CELLS=()
for defense in "${DEFENSES[@]}"; do
  eval "wanted=\$LEVELS_$defense"
  echo "  $defense: $wanted"
  for level in $wanted; do
    CELLS+=("$defense $level")
  done
done
echo "  ${#CELLS[@]} defended cells + nop"
echo

run nop mid1

# shuffled so several client containers can share a dataset without colliding
printf '%s\n' "${CELLS[@]}" | shuf | while read -r defense level; do
  run "$defense" "$level"
done
