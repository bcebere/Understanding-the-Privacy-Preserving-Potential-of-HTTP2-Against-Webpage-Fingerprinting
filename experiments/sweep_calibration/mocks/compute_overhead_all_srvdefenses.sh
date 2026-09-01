#!/usr/bin/env bash
# All 18 server-defense overhead cells, serially.
#   ./run_all_srvdefenses.sh <server_ip> [pages]
# Servers must already be up (./start_servers.sh on the server container).
# Serial on purpose: dT is wall-clock, so parallel cells inflate each other.
set -u

IP="$1"; PAGES="${2:-25}"
DS_DIR="$(basename "$PWD")"
DATASET="${DS_DIR%%_*}"

# first-party domain per dataset, for h2ps (1st-party-only by design)
case "$DS_DIR" in
  *amazon*) FIRST_PARTY="www.amazon.com" ;;
  *bbc*)    FIRST_PARTY="www.bbc.com" ;;
  *reddit*) FIRST_PARTY="www.reddit.com" ;;
  *udemy*)  FIRST_PARTY="www.udemy.com" ;;
  *wiki*)   FIRST_PARTY="en.wikipedia.org" ;;
  *) echo "unknown dataset $DS_DIR; set FIRST_PARTY by hand" >&2; exit 1 ;;
esac

DEFENSES=(alpaca tamaraw h2ps)
LEVELS=(vlow low lomid mid1 mid2 high vhigh vvhigh)

echo "dataset $DS_DIR (id $DATASET)   1st party: $FIRST_PARTY"
echo

for di in "${!DEFENSES[@]}"; do
  for li in "${!LEVELS[@]}"; do
    d="${DEFENSES[$di]}"; l="${LEVELS[$li]}"
    port=$((9000 + di * 100 + li * 10 + DATASET))

    # h2ps defends only the first party; the others are measured at "all"
    if [ "$d" = "h2ps" ]; then
      target="$FIRST_PARTY"
    else
      target="all"
    fi

    if ! timeout 2 bash -c "exec 3<>/dev/tcp/$IP/$port" 2>/dev/null; then
      echo "!!! $port  $d $l  server not reachable, skipping"
      continue
    fi

    bash ./compute_overhead_srvdefenses.sh "$IP" "$port" "$d" "$l" "$target" "$PAGES"
  done
done

WORKSPACE="/http2/experiments/sweep_calibration/${DS_DIR}/overhead"
echo
python ./calibrate.py --workspace "$WORKSPACE" --baseline nop
