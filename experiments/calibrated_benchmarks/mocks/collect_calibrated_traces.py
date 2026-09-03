#!/usr/bin/env python3
"""500-trace collection for the main tables.

    python3 collect_main.py client <dst_ip> <dst_port> <ifname> [--repeats 500]
    python3 collect_main.py server <server_ip> <ifname> [--repeats 500]

Levels and placements come from main_table_config.py.  Cell order is shuffled
so several containers can share a dataset; collect_traces.py skips captures
that already exist, so overlap is safe.
"""

import argparse
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

from main_table_config import cells, port


def stamp():
    return time.strftime("%F %T")


def reachable(ip, tcp_port, timeout=2):
    try:
        with socket.create_connection((ip, int(tcp_port)), timeout):
            return True
    except OSError:
        return False


def collect(tag, extra, repeats, ip, tcp_port, iface, common=()):
    print(f"=== {tag}  {stamp()}", flush=True)
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "collect_traces.py"),
        "--dst_ip",
        str(ip),
        "--dst_port",
        str(tcp_port),
        "--ifname",
        iface,
        "--tag",
        tag,
        "--repeats",
        str(repeats),
        *common,
        *extra,
    ]
    rc = subprocess.run(cmd).returncode
    print(f"=== {tag} {'done' if rc == 0 else 'FAILED'}  {stamp()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("side", choices=["client", "server"])
    parser.add_argument("ip")
    parser.add_argument(
        "rest", nargs="+", help="client: <dst_port> <ifname>;  server: <ifname>"
    )
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument(
        "--dataset", default=None, help="dataset name (default: current directory)"
    )
    parser.add_argument(
        "--workspace", default=None, help="workspace root passed to collect_traces.py"
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="server replicas per (defense, level); each worker "
        "picks the first reachable one, starting from a "
        "random offset so they spread out",
    )
    args = parser.parse_args()

    global DATASET
    DATASET = args.dataset or Path.cwd().name
    os.environ["WF_DATASET"] = DATASET
    plan = [c for c in cells(DATASET) if c[0] == args.side]
    random.shuffle(plan)

    print(
        f"dataset {DATASET}   side {args.side}   "
        f"repeats {args.repeats}   {len(plan)} cells\n"
    )

    common = ["--dataset", DATASET]
    if args.workspace:
        common += ["--workspace", args.workspace]

    if args.side == "client":
        if len(args.rest) != 2:
            sys.exit("client needs <dst_port> <ifname>")
        tcp_port, iface = args.rest
        for _, defense, level, tag, _ in plan:
            collect(
                tag,
                ["--defense", defense, "--level", level],
                args.repeats,
                args.ip,
                tcp_port,
                iface,
                common,
            )
    else:
        if len(args.rest) != 1:
            sys.exit("server needs <ifname>")
        iface = args.rest[0]
        for _, defense, level, tag, target in plan:
            base = port(DATASET, defense, level)
            # start from a random replica so concurrent workers do not all
            # pile onto replica 0
            offset = random.randrange(args.replicas)
            tcp_port = None
            for k in range(args.replicas):
                cand = base + ((offset + k) % args.replicas) * 1000
                if reachable(args.ip, cand):
                    tcp_port = cand
                    break
            if tcp_port is None:
                print(f"!!! {base}  {tag}  no replica reachable, skipping", flush=True)
                continue
            collect(
                tag,
                ["--defense", "nop", "--request_server_defense", target],
                args.repeats,
                args.ip,
                tcp_port,
                iface,
                common,
            )


if __name__ == "__main__":
    main()
