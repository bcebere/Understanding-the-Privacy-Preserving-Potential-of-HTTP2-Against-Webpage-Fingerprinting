docker stop "${1}"
docker rm -f "${1}"

docker run --log-opt max-size=10m --log-opt max-file=3 --name "$1" \
    -v $PWD/experiments_v2:/http2 -v $PWD/crawlers:/tls -v /data:/data -v $PWD/../wfaudit:/wfaudit -tid  --entrypoint /bin/bash http2_datasets

docker exec -it "$1" /bin/bash -c 'cd /wfaudit; pip install -e .;'
