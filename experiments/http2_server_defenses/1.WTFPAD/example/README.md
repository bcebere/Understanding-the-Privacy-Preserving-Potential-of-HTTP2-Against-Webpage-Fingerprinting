# Example usage

1. Start server with the WTF-PAD defense
```bash
bash ./run_server.sh 9999
```

2. Run client and collect the traces
```bash
bash ./run_client.sh 127.0.0.1 9999 lo
```

3. The collected traces are in the `traces/` folder. You can use this in the `wfaudit` tool to measure the information leakage.

4. For the best results, isolate the client and server using Docker container. See [the docker image](../../../docker_image).
