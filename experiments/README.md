# Experiments helpers
Helpers scripts for creating the traces and datasets required for the  `Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting` paper.

## Pre-requirements

### Docker images
We use docker images to isolate the client and server. The scripts for building and running the container are available [here](./docker_image).
The build create an image `http2_datasets`, which we use for both server and client sides.

### Creating TLS keys for the server
In order to enable TLS usage on the server, use the following snippet to create TLS key and certificate.

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

## Demo for simulating the HTTP2 datasets
[./demo](demo) illustrates how to collect traces using a toy dataset (3 pages), and how simulate various defenses (server or client side).
For full datasets, replace the `data` folder with the contents (`bin`, `client_trace`, `server_trace`) of one of the datasets from [../datasets/](../datasets/).

- Start the Docker containers
```bash
# Start client and server Docker containers

bash docker_image/run_container.sh http2_server

bash docker_image/run_container.sh http2_client
```

- Connect to the server Docker container and navigate to the demo folder
```bash
docker exec -it http2_server /bin/bash

ifconfig # Get the SERVER_IP, which will be used by the client from the other container

# Navigate to the demo folder
cd /experiments/demo

# Start a basic server on port 9999
bash ./run_server_dummy.sh 9999
```

- Connect to the client Docker container and test the connection
```bash
docker exec -it http2_client /bin/bash

# Navigate to the demo folder
cd /experiments/demo

# Connect a basic client to SERVER_IP and port 9999. This should finish without any errors.
python ./client_test.py --dst_ip $SERVER_IP --dst_port 9999

# Collect traces
rm -rf workspace
bash ./run_client_dummy.sh $SERVER_IP 9999 eth0

# If successful, this script will store in "workspace/traces" the PCAPs for 3 unique pages, 5 repeats per page.
# The "workspace/traces" path can be further passed to wfaudit -> "process_raw_pcaps" in the "traces" parameter.
```


### Client defenses
On the server side, we use always use the same implementation, provided using `bash ./run_server_dummy 9999`
On the client side, run

- HTTPOS defense
```bash
rm -rf workspace
bash ./run_cldefense_httpos.sh 172.17.0.2 9999 eth0
```

- FRONT defense
```bash
rm -rf workspace
bash ./run_cldefense_front.sh 172.17.0.2 9999 eth0
```

- Tamaraw defense
```bash
rm -rf workspace
bash ./run_cldefense_tamaraw.sh 172.17.0.2 9999 eth0
```

- H2PC defense
```bash
rm -rf workspace
bash ./run_client_modsnoise.sh 172.17.0.2 9999 eth0
```


### Server defenses
