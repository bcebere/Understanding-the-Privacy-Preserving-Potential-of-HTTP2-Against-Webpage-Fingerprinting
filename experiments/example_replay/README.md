# Replaying websites with various defenses

## Website content: `data`

Using the browser traces for a dataset, extract/copy the corresponding `data` folder into the current directory.

The `data` folder contains the webpage content used by the client and server to replay the websites under the evaluated defenses and create the resulting PCAP traces.

## Creating TLS keys for the server

To enable TLS on the server, use the following commands to create a private key and self-signed certificate:

```bash
# Directory to store keys
KEY_DIR="keys"
mkdir -p "$KEY_DIR"

# File paths
KEY_FILE="$KEY_DIR/key.pem"
CERT_FILE="$KEY_DIR/cert.pem"

# Generate private key
openssl genrsa -out "$KEY_FILE" 2048

# Generate self-signed certificate
openssl req -new -x509 -key "$KEY_FILE" -out "$CERT_FILE" -days 365 \
  -subj "/C=US/ST=Test/L=Local/O=TestOrg/OU=Dev/CN=localhost"
```

## Running the client/server in Docker containers

To isolate the experiments, we use separate Docker containers for the server and client. The scripts for building and running the containers are available [here](../docker_image).

The build creates the `http2_datasets` image, which is used for both the server and client.

First, build the image:

```bash
bash build.sh
```

Then start two containers:

```bash
bash ./run_container.sh http2_server
bash ./run_container.sh http2_client
```

Connect to either container using:

```bash
docker exec -it http2_server /bin/bash
docker exec -it http2_client /bin/bash
```

The replay scripts are available inside the containers at `/experiments/example_replay`.

## Client-side defenses

In the server container, first determine the server IP address using `ifconfig`.

Next, navigate to `/experiments/example_replay` and start the server.

The server requires valid `data` and `keys` directories inside the `example_replay` directory.

Start the server using:

```bash
bash ./run_server.sh <SERVER_PORT>
```

Next, in the client container, navigate to `/experiments/example_replay` and replay the website content under the client-side defenses:

```bash
python3 collect_calibrated_traces.py client <SERVER_IP> <SERVER_PORT> eth0 \
  --dataset <DATASET> --repeats 500

# Example
python3 collect_calibrated_traces.py client 172.17.0.2 8888 eth0 \
  --dataset 4_udemy --repeats 1
```

The script replays the website content under the calibrated client-side defenses and saves the resulting traffic as PCAP files.

Once collection is complete, the PCAPs are available under:

```text
workspace/<DATASET>/<DEFENSE>/traces/
```

## Server-side defenses

In the server container, navigate to `/experiments/example_replay` and start the defended servers. Each defense instance runs on a separate port.

```bash
python3 start_defended_servers.py start --replicas 4 --dataset <DATASET>

# Check whether the servers started successfully
python3 start_defended_servers.py status --dataset <DATASET>

# Example
python3 start_defended_servers.py start --replicas 4 --dataset 4_udemy
```

If everything is working, the `status` command shows the ports assigned to the different defense configurations and replicas, for example:

```text
  9034   alpaca   mid1   r0  UP
  10034  alpaca   mid1   r1  UP
  11034  alpaca   mid1   r2  UP
  12034  alpaca   mid1   r3  UP
  9124   tamaraw  lomid  r0  UP
  10124  tamaraw  lomid  r1  UP
  11124  tamaraw  lomid  r2  UP
  12124  tamaraw  lomid  r3  UP
  9244   h2ps     mid2   r0  UP
  10244  h2ps     mid2   r1  UP
  11244  h2ps     mid2   r2  UP
  12244  h2ps     mid2   r3  UP
```

On the client side, start the replay and trace collection using:

```bash
python3 collect_calibrated_traces.py server <SERVER_IP> eth0 \
  --replicas 4 --dataset <DATASET>

# Example
python3 collect_calibrated_traces.py server 172.17.0.44 eth0 \
  --replicas 4 --dataset 4_udemy
```

For server-side defenses, each defense configuration is associated with a specific server port. `collect_calibrated_traces.py` handles the mapping between ports and defense configurations.

As with the client-side defenses, the resulting PCAPs are available under:

```text
workspace/<DATASET>/<DEFENSE>/traces/
```

## Creating the evaluation datasets

Once the PCAPs have been collected:

1. Convert the PCAP traces to CSV:

   ```bash
   python3 ./benchmark_process_1_parse_traces.py
   ```

   The parsed CSV files are written to:

   ```text
   workspace/<DATASET>/<DEFENSE>/tcp_repr/output_csv_single/
   ```

2. Convert the parsed traces into the evaluation datasets:

   ```bash
   python3 ./benchmark_process_2_create_datasets.py
   ```

   The resulting `deepsetraces` and `wefdetraces` datasets are written to the
   workspace.

The resulting `deepsetraces` and `wefdetraces` can be used with the [example benchmark](../example_benchmark) for ML and MI analysis.
