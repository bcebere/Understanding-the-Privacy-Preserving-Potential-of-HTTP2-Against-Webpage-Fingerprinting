# Example usage

1. Start server
```bash
bash ./run_server.sh 9999
```
The server configuration can be changed using the parameters
```
python ./server.py --dst_port $1 \
    --use_server_push 0 \
    --use_rnd_multiplexing 0 \
    --use_rnd_hpack 0 \
    --use_hints103 0

```

2. Run client with the HTTPOS defense
```bash
bash ./run_client.sh 127.0.0.1 9999 lo
```

3. The collected traces are in the `traces/` folder. You can use this in the `wfaudit` tool to measure the information leakage.

4. For the best results, isolate the client and server using Docker container. See [the docker image](../../../docker_image).
