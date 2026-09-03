#!/usr/bin/env python3
"""Start server-side defenses for the main tables, with replicas.

    python3 start_benchmark_servers.py start [--replicas 4]
    python3 start_benchmark_servers.py status
    python3 start_benchmark_servers.py stop
    python3 start_benchmark_servers.py ports

Only the levels main_table_config.py selects are started -- 3 per dataset,
not 18.  Each is replicated so several client containers can drive the same
(defense, level) without queueing behind one single-threaded asyncio process.

    port = base + replica * 1000
    base = 9000 + defense_idx*100 + level_idx*10 + dataset_id

Replica 0 keeps the calibration port, so anything already running is reused.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

from main_table_config import SERVER, SERVER_ORDER, port

LOGDIR = Path("logs/servers")


def replica_port(defense, level, replica):
    return port(DATASET, defense, level) + replica * 1000


def up(p, host="127.0.0.1", timeout=1):
    try:
        with socket.create_connection((host, p), timeout):
            return True
    except OSError:
        return False


def plan(replicas):
    for defense in SERVER_ORDER:
        level = SERVER[DATASET][defense]
        for r in range(replicas):
            yield defense, level, r, replica_port(defense, level, r)


def main():
    parser = ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status", "ports"])
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument(
        "--dataset", default=None, help="dataset name (default: current directory)"
    )
    args = parser.parse_args()

    global DATASET
    DATASET = args.dataset or Path.cwd().name

    if args.action == "ports":
        for defense, level, r, p in plan(args.replicas):
            print(p, defense, level, f"replica{r}")
        return

    if args.action == "stop":
        killed = 0
        for defense, level, r, p in plan(args.replicas):
            out = subprocess.run(
                ["pgrep", "-f", f"--dst_port {p}"], capture_output=True, text=True
            ).stdout.split()
            for pid in out:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    killed += 1
                except OSError:
                    pass
        print(f"stopped {killed} servers for {DATASET}")
        return

    if args.action == "status":
        live = 0
        for defense, level, r, p in plan(args.replicas):
            state = "UP" if up(p) else "DOWN"
            live += state == "UP"
            log = LOGDIR / f"{defense}_{level}_r{r}.log"
            extra = "" if state == "UP" else f"   (see {log})"
            print(f"  {p:<6d} {defense:<8s} {level:<6s} r{r}  {state}{extra}")
        total = len(list(plan(args.replicas)))
        print(f"  {live}/{total} up  ({DATASET})")
        return

    LOGDIR.mkdir(parents=True, exist_ok=True)
    started = 0
    for defense, level, r, p in plan(args.replicas):
        if up(p):
            print(f"skip  {p}  {defense} {level} r{r}  (already up)")
            continue
        log = open(LOGDIR / f"{defense}_{level}_r{r}.log", "w")
        subprocess.Popen(
            [
                "bash",
                str(Path(__file__).parent / "run_server_level.sh"),
                str(p),
                defense,
                level,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"start {p}  {defense} {level} r{r}")
        started += 1

    if started:
        time.sleep(3)
    print()
    sys.argv = [
        sys.argv[0],
        "status",
        "--replicas",
        str(args.replicas),
        "--dataset",
        DATASET,
    ]
    main()


if __name__ == "__main__":
    main()
