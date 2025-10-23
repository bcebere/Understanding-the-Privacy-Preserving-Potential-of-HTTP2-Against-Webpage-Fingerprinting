# Experiments helpers
Helpers scripts for creating the traces and datasets required for the  `Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting` paper.

## Docker image
We use docker images to isolate the client and server. The scripts for building and running the container are available [here](./docker_image).

## Simulating the HTTP2 datasets

### Creating TLS keys

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
