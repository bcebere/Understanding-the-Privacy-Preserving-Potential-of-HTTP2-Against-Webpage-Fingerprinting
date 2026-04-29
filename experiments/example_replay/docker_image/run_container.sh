docker stop "${1}"
docker rm -f "${1}"

docker run --log-opt max-size=10m --log-opt max-file=3 --name "$1" \
    -v $PWD/:/experiments -v $PWD/../../../wfaudit:/wfaudit -v $PWD/../../../h2deflib:/h2deflib -tid  --entrypoint /bin/bash http2_datasets

docker exec -it "$1" /bin/bash -c 'cd /wfaudit; pip install -e .;'
docker exec -it "$1" /bin/bash -c 'cd /h2deflib; pip install -e .;'
