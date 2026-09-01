#!/usr/bin/env bash
# Collection progress per cell.
#   ./track_calibration_collect_status.sh [expected_total] [planned_cells]
#
# Once a cell is finished, output_csv_single is archived to
# <cell>_rawtraces.tar.zst and the folder is removed.  Counting members of a
# 50k-entry archive is slow, so the count is cached beside it in
# <cell>_rawtraces.count and reused.
set -u

EXPECTED="${1:-10000}"
PLANNED="${2:-45}"
DS="$(basename "$PWD")"
CAT="$(basename "$(dirname "$PWD")")"
RESULTS="${RESULTS:-/http2/experiments/$CAT/$DS/results}"

count_csv () {   # count_csv <cell_dir>
  local d="$1" cell single archive cache n
  cell=$(basename "$d")
  single="$d/tcp_repr/output_csv_single"

  if [ -d "$single" ]; then
    find "$single" -name 'temporal_data_*.csv' 2>/dev/null | wc -l
    return
  fi

  archive="$d/tcp_repr/${cell}_rawtraces.tar.zst"
  [ -f "$archive" ] || archive="$d/tcp_repr/${cell}_rawtraces.tar.gz"
  if [ ! -f "$archive" ]; then
    echo 0
    return
  fi

  cache="${archive%.tar.*}.count"
  if [ -f "$cache" ]; then
    cat "$cache"
    return
  fi

  if [[ "$archive" == *.zst ]]; then
    n=$(tar --use-compress-program=unzstd -tf "$archive" 2>/dev/null \
        | grep -c 'temporal_data_.*\.csv')
  else
    n=$(tar -tzf "$archive" 2>/dev/null | grep -c 'temporal_data_.*\.csv')
  fi
  echo "$n" > "$cache" 2>/dev/null
  echo "$n"
}

printf "%-18s %8s %8s %8s %7s %5s  %s\n" CELL PCAP CSV TOTAL PCT SRC SIZE
grand=0; started=0; done_cells=0; archived=0
for d in "$RESULTS"/*/; do
  [ -d "$d/traces" ] || continue
  cell=$(basename "$d")
  pcap=$(find "$d/traces" -name '*.pcap' 2>/dev/null | wc -l)
  csv=$(count_csv "$d")
  if [ -d "$d/tcp_repr/output_csv_single" ]; then src="dir"; else src="tar"; archived=$((archived+1)); fi
  size=$(du -sh "$d" 2>/dev/null | cut -f1)
  tot=$((pcap + csv))
  grand=$((grand + tot)); started=$((started + 1))
  [ "$tot" -ge "$EXPECTED" ] && done_cells=$((done_cells + 1))
  printf "%-18s %8d %8d %8d %6d%% %5s  %s\n" "$cell" "$pcap" "$csv" "$tot" \
         $((tot * 100 / EXPECTED)) "$src" "$size"
done

target=$((PLANNED * EXPECTED))
pct=$((grand * 100 / target))
gb=$(du -sb "$RESULTS" 2>/dev/null | cut -f1)

echo
printf "cells    %d/%d complete, %d started, %d not begun, %d archived\n" \
       "$done_cells" "$PLANNED" "$started" "$((PLANNED - started))" "$archived"
printf "captures %d/%d  (%d%%)  -- %d remaining\n" \
       "$grand" "$target" "$pct" "$((target - grand))"
if [ "$grand" -gt 0 ] && [ -n "$gb" ]; then
  printf "disk     %s used, ~%d GB projected at completion\n" \
         "$(du -sh "$RESULTS" 2>/dev/null | cut -f1)" \
         "$((gb / 1000000000 * target / grand))"
fi

# throughput from the oldest capture still on disk; archived cells are
# excluded, so this is the rate of what is still in flight
oldest=$(find "$RESULTS" -name '*.pcap' -o -name 'temporal_data_*.csv' 2>/dev/null \
         | head -20000 | xargs -r stat -c %Y 2>/dev/null | sort -n | head -1)
if [ -n "$oldest" ] && [ "$grand" -gt 0 ]; then
  el=$(( $(date +%s) - oldest ))
  if [ "$el" -gt 60 ]; then
    rate=$(( grand * 3600 / el ))
    [ "$rate" -gt 0 ] && printf "rate     %d captures/h  -> ~%d h (%d days) left\n" \
      "$rate" "$(( (target - grand) / rate ))" "$(( (target - grand) / rate / 24 ))"
  fi
fi

echo
echo "unprocessed pcaps: $(find "$RESULTS" -name '*.pcap' 2>/dev/null | wc -l) ($(du -sh "$RESULTS" 2>/dev/null | cut -f1) total)"
echo "in flight (.part): $(find "$RESULTS" -name '*.pcap.*.part' 2>/dev/null | wc -l)"
