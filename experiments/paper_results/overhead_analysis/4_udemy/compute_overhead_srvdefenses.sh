WORKSPACE="workspace"
DEF="$3"

if [ -z "$DEF" ]; then
  echo "Error: missing server defense argument (\$3)" >&2
  exit 1
fi

#nohup bash ./run_client_srvdefs.sh 172.17.0.2 8181 eth0 all > /dev/null 2>&1&
#nohup bash ./run_client_srvdefs.sh 172.17.0.2 8261 eth0 www.udemy.com > /dev/null 2>&1&
#nohup bash ./run_client_srvdefs.sh 172.17.0.2 8321 eth0 challenges.cloudflare.com > /dev/null 2>&1&


declare -A SCENARIO_TARGETS=(
  ["all"]="all"
  ["1st"]="www.udemy.com"
  ["3rd_1"]="challenges.cloudflare.com"
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
