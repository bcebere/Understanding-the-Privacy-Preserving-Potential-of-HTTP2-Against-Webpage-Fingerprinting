#!/usr/bin/env bash
bash "$(dirname "$0")"/run_server_level.sh "$1" "${2:-nop}" "${3:-mid1}"
