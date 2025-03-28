# stdlib
from pathlib import Path

# third party
import pyshark

# wfaudit absolute
from wfaudit.processing.flow_session import FlowSession


def process_capture(
    name: str,
    capture,
    workspace: Path = Path("workspace"),
    with_certificates: bool = False,
    with_dns: bool = False,
    buffer_tcp: bool = True,
) -> FlowSession:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    session = FlowSession(
        capture,
        name=name,
        workspace=workspace,
        with_certificates=with_certificates,
        with_dns=with_dns,
        buffer_tcp=buffer_tcp,
    )

    return session


def process_pcap(
    input_file: Path,
    workspace: Path = Path("workspace"),
    with_certificates: bool = False,
    with_dns: bool = False,
    buffer_tcp: bool = True,
) -> FlowSession:
    input_file = Path(input_file)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    name = input_file.stem

    filtered_cap = pyshark.FileCapture(
        input_file,
        display_filter="tls",
    )

    session = process_capture(
        name=name,
        capture=filtered_cap,
        workspace=workspace,
        with_certificates=with_certificates,
        with_dns=with_dns,
        buffer_tcp=buffer_tcp,
    )

    filtered_cap.close()

    return session
