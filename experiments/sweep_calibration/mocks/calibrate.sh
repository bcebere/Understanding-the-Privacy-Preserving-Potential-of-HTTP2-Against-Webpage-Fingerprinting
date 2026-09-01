#!/usr/bin/env bash

SCENARIO="`basename $PWD`"
WORKSPACE="/http2/experiments/sweep_calibration/${SCENARIO}/overhead"

export WF_DATASET="${WF_DATASET:-$(basename "$PWD")}"
mkdir -p "$WORKSPACE"

echo "WORKSPACE $WORKSPACE"
python ./calibrate.py --workspace "$WORKSPACE" --baseline nop
