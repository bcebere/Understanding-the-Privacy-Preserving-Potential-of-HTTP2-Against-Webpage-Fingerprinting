#!/usr/bin/env bash
# Start all 18 server-defense cells, one per port, for this dataset.
#   ./start_servers.sh [start|stop|status] [dataset_id]
# Port = 9000 + defense*100 + level*10 + dataset
#   defense: alpaca 0, tamaraw 1, h2ps 2
#   level:   vlow 0, low 1, lomid 2, mid1 3, mid2 4, high 5
#   dataset: trailing digit (1 amazon .. 5 wiki), taken from the dir name
set -u

ACTION="${1:-start}"
DS_DIR="$(basename "$PWD")"
DATASET="${2:-${DS_DIR%%_*}}"
LOGDIR="logs/servers"

DEFENSES=(alpaca tamaraw h2ps)
LEVELS=(vlow low lomid mid1 mid2 high vhigh vvhigh)

port_for () {  # port_for <defense_idx> <level_idx>
  echo $((9000 + $1 * 100 + $2 * 10 + DATASET))
}

case "$ACTION" in
  start)
    mkdir -p "$LOGDIR"
    for di in "${!DEFENSES[@]}"; do
      for li in "${!LEVELS[@]}"; do
        d="${DEFENSES[$di]}"; l="${LEVELS[$li]}"
        p=$(port_for "$di" "$li")
        if timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$p" 2>/dev/null; then
          echo "skip  $p  $d $l  (already up)"
          continue
        fi
        nohup bash ./run_server_level.sh "$p" "$d" "$l" \
            > "$LOGDIR/${d}_${l}.log" 2>&1 &
        echo "start $p  $d $l"
      done
    done
    sleep 3
    echo
    bash "$0" status "$DATASET"
    ;;

  stop)
    pkill -f "server_with_defs.py --dst_port 9[0-9][0-9]${DATASET}" 2>/dev/null
    pkill -f "server_simple.py --dst_port 9[0-9][0-9]${DATASET}" 2>/dev/null
    echo "stopped servers for dataset $DATASET"
    ;;

  status)
    up=0
    for di in "${!DEFENSES[@]}"; do
      for li in "${!LEVELS[@]}"; do
        d="${DEFENSES[$di]}"; l="${LEVELS[$li]}"
        p=$(port_for "$di" "$li")
        if timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$p" 2>/dev/null; then
          printf "  %-5s %-8s %-6s UP\n" "$p" "$d" "$l"; up=$((up+1))
        else
          printf "  %-5s %-8s %-6s DOWN   (see %s/%s_%s.log)\n" \
                 "$p" "$d" "$l" "$LOGDIR" "$d" "$l"
        fi
      done
    done
    echo "  $up/18 up  (dataset $DATASET)"
    ;;

  ports)
    for di in "${!DEFENSES[@]}"; do
      for li in "${!LEVELS[@]}"; do
        echo "$(port_for "$di" "$li") ${DEFENSES[$di]} ${LEVELS[$li]}"
      done
    done
    ;;

  *) echo "usage: $0 [start|stop|status|ports] [dataset_id]" >&2; exit 1 ;;
esac
