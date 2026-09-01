#!/usr/bin/env bash
# PCAP collection for the server-defense cells.
#   ./collect_all_srvdefenses.sh <server_ip> <ifname> [repeats]
# Servers must already be up (./start_servers.sh on the server container).
#
# Per-defense level lists: alpaca and tamaraw stop at mid1.  Their mid2/high
# cells cost 15-13x page load time and ~200 GB each -- outside any deployable
# range, so they are not collected.  h2ps is cheap at every level and keeps
# all six, which is where its advantage over the padding defenses shows.
#
# Cell order is shuffled so several client containers can work the same
# dataset without colliding; collect_traces.py skips captures that already
# exist and writes via a pid-suffixed temp file, so overlap is safe.
# Run one client per container: concurrent captures on one interface contend
# for CPU and perturb inter-arrival times, which are a leakage feature.
set -u

IP="$1"; IFACE="$2"; REPEATS="${3:-100}"
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

# index order must match server_defenses/levels.py, since the port encodes it
DEFENSES=(alpaca tamaraw h2ps)
LEVELS=(vlow low lomid mid1 mid2 high vhigh vvhigh)

# which levels to collect, per defense
LEVELS_alpaca="vlow low lomid mid1"
LEVELS_tamaraw="vlow low lomid"
LEVELS_h2ps="vlow low lomid mid1 mid2 high vhigh"

echo "dataset $DS_DIR (id $DATASET)   1st party: $FIRST_PARTY   repeats: $REPEATS"
for d in "${DEFENSES[@]}"; do
  eval "wanted=\$LEVELS_$d"
  echo "  $d: $wanted"
done
echo

# shuffle the index pairs, not the arrays: the port depends on both indices
CELLS=()
for di in "${!DEFENSES[@]}"; do
  d="${DEFENSES[$di]}"
  eval "wanted=\$LEVELS_$d"
  for li in "${!LEVELS[@]}"; do
    case " $wanted " in
      *" ${LEVELS[$li]} "*) CELLS+=("$di $li") ;;
    esac
  done
done

echo "${#CELLS[@]} cells to collect"
echo

printf '%s\n' "${CELLS[@]}" | shuf | while read -r di li; do
  d="${DEFENSES[$di]}"; l="${LEVELS[$li]}"
  port=$((9000 + di * 100 + li * 10 + DATASET))

  if [ "$d" = "h2ps" ]; then
    target="$FIRST_PARTY"
    tag="srv${d}1p_${l}"
  else
    target="all"
    tag="srv${d}_${l}"
  fi

  if ! timeout 2 bash -c "exec 3<>/dev/tcp/$IP/$port" 2>/dev/null; then
    echo "!!! $port  $d $l  server not reachable, skipping"
    continue
  fi

  echo "=== $tag  port $port  target $target  $(date '+%F %T')"
  python ./collect_traces.py --dst_ip "$IP" --dst_port "$port" --ifname "$IFACE" \
      --defense nop --request_server_defense "$target" \
      --tag "$tag" --repeats "$REPEATS"
  echo "=== $tag done  $(date '+%F %T')"
done
