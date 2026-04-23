WORKSPACE="workspace"
DEF="h2srv"

target="www.reddit.com"
scenario="1st"
csv_file="${WORKSPACE}/ovh_${DEF}_${scenario}.csv"

if [ ! -f "$csv_file" ]; then
  echo "Evaluate $DEF $scenario $target"
  python ./approximate_overhead.py \
	--dst_ip "$1" \
	--dst_port "$2" \
	--request_server_defense "$target" \
	--scenario "${DEF}_${scenario}"
fi
