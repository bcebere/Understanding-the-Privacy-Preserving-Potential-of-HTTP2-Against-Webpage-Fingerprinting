WORKSPACE="workspace"
DEF="$3"

# Exit early instead of continuing with a broken state
if [ -z "$DEF" ]; then
  echo "Error: missing server defense argument (\$3)" >&2
  exit 1
fi

# Associative array replaces the if/elif chain
declare -A SCENARIO_TARGETS=(
  ["all"]="all"
  ["1st"]="www.amazon.com"
  ["3rd_1"]="m.media-amazon.com"
  ["3rd_2"]="images-na.ssl-images-amazon.com"
)

for scenario in "${!SCENARIO_TARGETS[@]}"; do
  csv_file="${WORKSPACE}/ovh_${DEF}_${scenario}.csv"

  if [ ! -f "$csv_file" ]; then
    target="${SCENARIO_TARGETS[$scenario]}"
    echo "Evaluate $DEF $scenario $target"
    python ./approximate_overhead.py \
      --dst_ip "$1" \
      --dst_port "$2" \
      --request_server_defense "$target" \
      --scenario "${DEF}_${scenario}"
  fi
done
