"""
"""

import json
import os
import random
import time
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from h2deflib import Request, run_test_case
from tqdm import tqdm

# ====================================================================== #
# Setup                                                                   #
# ====================================================================== #


def _parse_args():
    p = ArgumentParser()
    p.add_argument("-dst_ip", "--dst_ip", required=True)
    p.add_argument("-dst_port", "--dst_port", required=True)
    p.add_argument(
        "-ifname", "--ifname", required=True, help="Interface for packet capture"
    )
    p.add_argument("-repeats", "--repeats", default=500, type=int)
    p.add_argument("-subpage_limit", "--subpage_limit", default=100, type=int)
    p.add_argument(
        "-defense",
        "--defense",
        default="nop",
        help="Defense name passed to h2deflib.get_defense",
    )
    p.add_argument(
        "-request_server_defense",
        "--request_server_defense",
        default=None,
        help="Connection id to request server-side defense for "
        "('all' for every connection)",
    )

    def _str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1", "true", "yes", "t")

    p.add_argument("-capture", "--capture", default=True, type=_str2bool)

    return p.parse_args()


DATA_PATH = Path("data")
SRV_DB_PATH = DATA_PATH / "server_trace"
CLIENT_DB_PATH = DATA_PATH / "client_trace"

WORKSPACE = Path("workspace")
TRACE_PATH = WORKSPACE / "traces"
OUTPUT_CSV = WORKSPACE / "tcp_repr/output_csv_single"
STARTUP_WAIT_SEC = 0.2


# ====================================================================== #
# Helpers                                                                 #
# ====================================================================== #


def _domain(url: str) -> str:
    return urlparse(url).netloc


def prepare_requests(
    raw_requests: List[dict],
    request_server_defense: str = None,
) -> Tuple[Dict[str, List[Request]], Dict[str, bool], bool]:
    """
    Convert the raw JSON records into h2deflib-compatible Request objects
    grouped by ``connection_id``. Also computes, for each connection,
    whether server-side defenses should be activated (based on the
    ``request_server_defense`` CLI flag).
    """
    out: Dict[str, List[Request]] = {}
    defenses: Dict[str, bool] = {}
    has_binary = False

    binary_exts = ("gif", "png", "svg", "jpg", "jpeg", "pdf", "webp")

    for ridx, req in enumerate(raw_requests):
        testcase = req["label"]
        with open(SRV_DB_PATH / f"{testcase}.json") as f:
            server_db = json.load(f)

        full_url = req["url"]
        conn_id = _domain(full_url)
        path = req["url_local"]

        if path.lower().split("?")[0].endswith(binary_exts):
            has_binary = True

        file_path = server_db[path]["body_path"]
        expected_size = os.path.getsize(file_path)

        if conn_id not in out:
            out[conn_id] = []
            if request_server_defense == "all":
                defenses[conn_id] = True
            elif request_server_defense == conn_id:
                defenses[conn_id] = True
            else:
                defenses[conn_id] = False

        # First request on each connection gets a small staggered delay to
        # make it look like the browser's initial roundtrip.
        delay = 0.01 * ridx if not out[conn_id] else 0.001

        out[conn_id].append(
            Request(
                path=path,
                label=testcase,
                data={},
                headers={},
                delay=delay,
                expected_size=expected_size,
                connection_id=conn_id,
            )
        )

    if (
        request_server_defense is not None
        and request_server_defense not in out
        and request_server_defense != "all"
    ):
        print(
            f"Server defense requested for {request_server_defense}, "
            f"but I only have {list(out.keys())}"
        )
        return {}, {}, False

    return out, defenses, has_binary


def _list_valid_testcases() -> List[str]:
    paths = glob(str(CLIENT_DB_PATH / "*"))
    valid = []
    for path in tqdm(paths, desc="scanning testcases"):
        raw = json.load(open(path))
        if len(raw) < 3:
            continue
        # Tag every record with the testcase label (used later for DB lookup)
        label = Path(path).stem
        for r in raw:
            r["label"] = label
        reqs, _, _ = prepare_requests(deepcopy(raw))
        if reqs:
            valid.append(path)
    return valid


# ====================================================================== #
# Main                                                                    #
# ====================================================================== #


def main():
    args = _parse_args()

    if args.capture:
        from scapy.config import conf
        from scapy.sendrecv import AsyncSniffer
        from scapy.utils import wrpcap

        conf.route_autoload = False
        conf.route6_autoload = False
        conf.bufsize = 50 * 1024 * 1024

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.mkdir(parents=True, exist_ok=True)

    print(
        f"[experimental_client] defense={args.defense} "
        f"server_defense={args.request_server_defense}"
    )

    valid_paths = _list_valid_testcases()
    run_paths = valid_paths[: args.subpage_limit]
    random.shuffle(run_paths)
    repeat_order = random.sample(range(args.repeats), k=args.repeats)

    for repeat in repeat_order:
        for tc_path in tqdm(run_paths, desc=f"repeat {repeat}"):
            label = Path(tc_path).stem
            raw = json.load(open(tc_path))
            assert len(raw) >= 3
            for r in raw:
                r["label"] = label

            requests_by_conn, server_defenses, _ = prepare_requests(
                deepcopy(raw),
                request_server_defense=args.request_server_defense,
            )
            assert requests_by_conn

            save_label = f"test_{label}_{repeat}"
            trace_file = TRACE_PATH / f"{save_label}.pcap"
            out_csv = OUTPUT_CSV / f"temporal_data_{save_label}.csv"
            if out_csv.exists() or trace_file.exists():
                continue

            tracer = None
            if args.capture:
                # Start packet capture
                tracer = AsyncSniffer(iface=args.ifname)
                tracer.start()
                for retry in range(10):
                    if not hasattr(tracer, "stop_cb"):
                        time.sleep(STARTUP_WAIT_SEC)
                    else:
                        break

            # Run the test case through h2deflib
            run_test_case(
                server_ip=args.dst_ip,
                server_port=int(args.dst_port),
                requests_by_connection=requests_by_conn,
                request_server_defenses=server_defenses,
                defense_name=args.defense,
            )

            if tracer is not None:
                # Persist capture
                network_trace = tracer.stop()
                with open(trace_file, "wb") as f:
                    wrpcap(f, network_trace)
                del network_trace, tracer


if __name__ == "__main__":
    main()
