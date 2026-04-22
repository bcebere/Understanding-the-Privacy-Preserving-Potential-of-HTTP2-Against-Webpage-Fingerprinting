WORKSPACE="workspace"

if [ ! -f ${WORKSPACE}/"ovh_baseline.csv" ]
then
  echo "Evaluate baseline"
  python ./approximate_overhead.py --dst_ip $1 --dst_port $2  --scenario "baseline"
fi

if [ ! -f ${WORKSPACE}/"ovh_h2pc.csv" ]
then
  echo "Evaluate h2pc"
  python ./approximate_overhead.py --dst_ip $1 --dst_port $2 --http2_all 1 --scenario "h2pc"
fi


for def in "front" "tamaraw" "httpos" "llama"
do

  if [ -f ${WORKSPACE}/"ovh_${def}.csv" ]
  then
    continue
  fi
  echo "Evaluate $def"
  python ./approximate_overhead.py --dst_ip $1 --dst_port $2 --defense "$def" --scenario "$def"
done
