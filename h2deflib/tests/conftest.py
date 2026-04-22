# future
from __future__ import annotations

# stdlib
import datetime
import random
from pathlib import Path
from typing import Tuple

import pytest

# third party
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def _make_self_signed_cert(tmp_path: Path) -> Tuple[Path, Path]:
    """Generate a throwaway TLS cert for the tests."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_file = tmp_path / "cert.pem"
    key_file = tmp_path / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


@pytest.fixture(scope="session")
def tls_cert(tmp_path_factory):
    """Generate one self-signed cert per test session."""
    tmp = tmp_path_factory.mktemp("tls")
    return _make_self_signed_cert(tmp)


@pytest.fixture(autouse=True)
def seeded_random():
    """Make the tests deterministic despite the server's use of ``random``."""
    random.seed(1234)
