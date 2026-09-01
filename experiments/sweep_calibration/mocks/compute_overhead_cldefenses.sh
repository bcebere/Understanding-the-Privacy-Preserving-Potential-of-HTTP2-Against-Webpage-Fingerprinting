#!/usr/bin/env bash

IP="$1"
PORT="$2"
PAGES="${3:-25}"

SCENARIO="`basename $PWD`"
WORKSPACE="/http2/experiments/sweep_calibration/${SCENARIO}/overhead"

export WF_DATASET="${WF_DATASET:-$(basename "$PWD")}"
mkdir -p "$WORKSPACE"

run () {
  local scen="$1"; shift
  if [ -f "$WORKSPACE/ovh_${scen}.csv" ]; then
    echo "skip  $scen"
    return
  fi
  echo "=== $scen"
  H2_VERBOSE=0 python ./approximate_overhead.py \
      --workspace "$WORKSPACE" --dst_ip "$IP" --dst_port "$PORT" --pages "$PAGES" --scenario "$scen" "$@"
}

run baseline   # needed as the latency denominator

for lvl in vlow low lomid mid1 mid2 high; do
  run "front_$lvl"   --defense front   --level "$lvl"
  run "tamaraw_$lvl" --defense tamaraw --level "$lvl"
  run "h2pc_$lvl"    --defense h2pc    --level "$lvl"
  run "httpos_$lvl"  --defense httpos  --level "$lvl"
  run "llama_$lvl"   --defense llama   --level "$lvl"
done

echo "WORKSPACE $WORKSPACE"
python ./calibrate.py --workspace "$WORKSPACE" --baseline nop
