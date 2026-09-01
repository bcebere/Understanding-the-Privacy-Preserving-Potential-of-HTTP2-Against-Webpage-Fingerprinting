#!/usr/bin/env bash
# Start a server-side defense at a given intensity.
#   ./run_server_level.sh <port> <alpaca|tamaraw|h2ps> <level>
set -u
PORT="$1"; DEF="$2"; LVL="$3"

case "$DEF" in
  alpaca)  exec python ./server_with_defs.py --dst_port "$PORT" --defense_alpaca 1  --level "$LVL" ;;
  tamaraw) exec python ./server_with_defs.py --dst_port "$PORT" --defense_tamaraw 1 --level "$LVL" ;;
  h2ps)    HPACK=$(python3 -c "from server_defenses.levels import params; print(params('h2ps','$LVL')['hpack'])")
           exec python ./server_simple.py --dst_port "$PORT" --level "$LVL" \
             --http2_batch 1 --http2_rnd_hints103 1 --http2_global_hints103 1 \
             --http2_rnd_out_window 1 --http2_rnd_delay 1 \
             --http2_hpack "$HPACK" --http2_rnd_server_push 0 ;;
  nop)     exec python ./server_simple.py --dst_port "$PORT" ;;
  *) echo "unknown defense $DEF" >&2; exit 1 ;;
esac
