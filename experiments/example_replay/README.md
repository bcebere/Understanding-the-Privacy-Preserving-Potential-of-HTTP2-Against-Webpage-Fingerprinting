# HTTP/2 Traffic Replay Examples

## Pre-requirements

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
[example_replay](./example_replay) illustrates how to replay browser traces, using various defenses (server or client side).
For full datasets, replace the `data` folder with the contents of one of the `browser_original_traces` (`bin`, `client_trace`, `server_trace`) from [the datasets repository](https://i62nextcloud.tm.kit.edu/public.php/dav/files/6ga8tgFyiXo4ZAf/?accept=zip).


### Client defenses
On the server side, we use always use the same implementation, provided using `./server_nop.sh`

Start the server using
```bash
bash ./server_nop.sh --dst_port 9999
```

On the client side, run

```bash
# Preparation
rm -rf workspace
SERVER_IP=... # from ifconfig from the server-side
SERVER_PORT=9999
```

- HTTPOS defense
```bash
bash ./client_tamaraw.sh --dst_ip $SERVER_IP --dst_port $SERVER_PORT --capture 0 # provide ifname also for capture=1
```

- FRONT defense
```bash
bash ./client_front.sh --dst_ip $SERVER_IP --dst_port $SERVER_PORT --capture 0 # provide ifname also for capture=1
```

- Tamaraw defense
```bash
bash ./client_tamaraw.sh --dst_ip $SERVER_IP --dst_port $SERVER_PORT --capture 0 # provide ifname also for capture=1
```

- H2PC defense
```bash
bash ./client_h2pc.sh --dst_ip $SERVER_IP --dst_port $SERVER_PORT --capture 0 # provide ifname also for capture=1
```


### Server defenses
On the server side, run:


- ALPACA defense
```bash
bash ./server_alpaca.sh --dst_port 9999
```

- TAMARAW defense
```bash
bash ./server_tamaraw.sh --dst_port 9999
```

- H2PS defense
```bash
bash ./server_h2ps.sh --dst_port 9999
```

On the client side, request defense for the 'en.wikipedia.org' server using
```bash
bash ./client_srvdefs.sh --dst_ip $SERVER_IP --dst_port 9999 --capture 0 --request_server_defense en.wikipedia.org
```
### Docker images (optional)
For isolating multiple experiments, we used docker containers for each the server and the client. The scripts for building and running the container are available [here](./docker_image).
The build create an image `http2_datasets`, which we use for both server and client sides.

- Start the Docker containers
```bash
# Start client and server Docker containers

bash docker_image/run_container.sh http2_server

bash docker_image/run_container.sh http2_client
```

- Connect to the server Docker container and navigate to the example folder
```bash
docker exec -it http2_server /bin/bash

ifconfig # Get the SERVER_IP, which will be used by the client from the other container

# Navigate to the demo folder
cd /experiments/example_replay

# Start a basic server on port 9999
bash ./nop_server_nop.sh --dst_port 9999
```

- Connect to the client Docker container and test the connection
```bash
docker exec -it http2_client /bin/bash

# Navigate to the demo folder
cd /experiments/example_replay/

# Connect a basic client to SERVER_IP and port 9999. This should finish without any errors.
bash ./client_nop.sh --dst_ip $SERVER_IP --dst_port 9999

# If successful, this script will store in "workspace/traces" the PCAPs for 3 unique pages, 500 repeats per page.
```
