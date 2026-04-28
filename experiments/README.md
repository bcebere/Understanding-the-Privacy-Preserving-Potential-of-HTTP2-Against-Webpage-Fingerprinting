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
[example_replay](./example_replay) illustrates how to replay browser traces, using various defenses (server or client side).
For full datasets, replace the `data` folder with the contents (`bin`, `client_trace`, `server_trace`) of one of the datasets from [website datasets](../datasets/).

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

# If successful, this script will store in "workspace/traces" the PCAPs for 3 unique pages, 150 repeats per page.
# The "workspace/traces" path can be further passed to wfaudit -> "process_raw_pcaps" in the "traces" parameter.
```


### Client defenses
On the server side, we use always use the same implementation, provided using `bash ./run_server_dummy 9999`
On the client side, run

```bash
# Preparation
rm -rf workspace
SERVER_IP=... # from ifconfig from the server-side
```

- HTTPOS defense
```bash
bash ./run_cldefense_httpos.sh $SERVER_IP 9999 eth0
```

- FRONT defense
```bash
bash ./run_cldefense_front.sh $SERVER_IP 9999 eth0
```

- Tamaraw defense
```bash
bash ./run_cldefense_tamaraw.sh $SERVER_IP 9999 eth0
```

- H2PC defense
```bash
bash ./run_client_h2pc.sh $SERVER_IP 9999 eth0
```


### Server defenses
On the server side, run:


- ALPACA defense
```bash
bash ./run_server_def_alpaca.sh 9999
```

- TAMARAW defense
```bash
bash ./run_server_def_tamaraw.sh 9999
```

- H2PS defense
```bash
bash ./run_server_def_h2ps.sh 9999
```

On the client side, request defense for the 'en.wikipedia.org' server using
```bash
bash ./run_client_srvdefs.sh $SERVER_IP 9999 eth0 en.wikipedia.org
```

### Dataset creation and security estimation
Once the data collection is done, create the evaluation datasets using `wfaudit`

```python
# Create evaluation datasets from the raw traces
from pathlib import Path
from wfaudit import (
    prepare_all_datasets,
    process_raw_pcaps,
)

workspace = Path("workspace")
pcaps = process_raw_pcaps(
    traces=workspace / "traces",
    workspace=workspace,
    unlink_after_processing=False,
)

prepare_all_datasets(
    workspace=workspace,
    n_websites=3,
    n_traces=150,
)
```

Next, we can use these dataset to evaluate the security of the scenario (ML classification, information leakage and feature importance):
```python
# Evaluate the security of the dataset.

# stdlib
from pathlib import Path

# wfaudit absolute
from wfaudit import (
    audit,
)

workspace = Path("workspace")
ml_output_folder = workspace / "eval_ml"
wefde_output_folder = workspace / "eval_wefde"
deepse_output = workspace / "eval_deepse/results.csv"
xai_output_folder = workspace / "eval_xai"

wefde_feats_folder = workspace / "output_features"
deepse_dataset = workspace / "output_deepse" / "real" / "dataset.npz"

scores = audit(
    # ML
    ml_output_folder=ml_output_folder,
    wefde_feats_folder=wefde_feats_folder,
    deepse_dataset=deepse_dataset,
    ml_arch_2D=["xgboost"],
    ml_arch_3D=[],
    # leakage
    wefde_output_folder=wefde_output_folder,
    deepse_output=deepse_output,
    # xai
    xai_output_folder=xai_output_folder,
)

print("ML scores ---> ", scores["ML"])
print("Leakage scores ---> ", scores["leakage"])
```
