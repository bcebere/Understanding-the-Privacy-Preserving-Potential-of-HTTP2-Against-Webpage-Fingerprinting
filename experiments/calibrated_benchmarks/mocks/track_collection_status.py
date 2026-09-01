#!/usr/bin/env python3
"""Collection and processing progress for the main tables.

    python3 track_collection_status.py [expected_traces]

Run from a dataset dir.  Expected cells come from main_table_config, so a cell
that has not started is reported rather than silently absent.

Once a cell is finished, output_csv_single is archived to
<cell>_rawtraces.tar.zst and the folder removed.  Counting members of a 50k
archive is slow, so the count is cached in <cell>_rawtraces.count.
"""

import subprocess
import sys
import time
from pathlib import Path

from main_table_config import cells

EXPECTED = int(sys.argv[1]) if len(sys.argv) > 1 else 500
PAGES = 100
TARGET = EXPECTED * PAGES

DATASET = Path.cwd().name
CATEGORY = Path.cwd().parent.name
RESULTS = Path(f"/http2/experiments/{CATEGORY}/{DATASET}/results")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def du_bytes(path):
    try:
        return int(run(["du", "-sb", str(path)]).split()[0])
    except Exception:
        return 0


def human(n):
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}P"


def csv_count(cell):
    """-> (count, source).  Reads the archive when the folder is gone."""
    single = cell / "tcp_repr/output_csv_single"
    if single.is_dir():
        return len(list(single.glob("temporal_data_*.csv"))), "dir"

    for suffix, lister in (
        (".tar.zst", ["tar", "--use-compress-program=unzstd", "-tf"]),
        (".tar.gz", ["tar", "-tzf"]),
    ):
        archive = cell / "tcp_repr" / f"{cell.name}_rawtraces{suffix}"
        if not archive.exists():
            continue
        cache = archive.with_name(f"{cell.name}_rawtraces.count")
        if cache.exists():
            try:
                return int(cache.read_text().strip()), "tar"
            except ValueError:
                pass
        out = run(lister + [str(archive)])
        n = sum(1 for line in out.splitlines() if "temporal_data_" in line)
        if n:
            try:
                cache.write_text(str(n))
            except OSError:
                pass
        return n, "tar"

    return 0, "-"


plan = [tag for _, _, _, tag, _ in cells(DATASET)]
rows, grand, done_cells, oldest, archived = [], 0, 0, None, 0

for tag in plan:
    cell = RESULTS / tag
    traces = cell / "traces"
    pcap = len(list(traces.glob("*.pcap"))) if traces.is_dir() else 0
    csv, source = csv_count(cell)
    if source == "tar":
        archived += 1

    total = pcap + csv
    grand += total
    if total >= TARGET:
        done_cells += 1

    ready = (cell / "tcp_repr/output_ml/X_1C.npy").exists()
    tcp = cell / "tcp_repr"
    scored = len(list(tcp.glob("eval_ml*/scores_rawts_*.bkp"))) if tcp.is_dir() else 0

    if traces.is_dir():
        for f in list(traces.glob("*.pcap"))[:200]:
            ts = f.stat().st_mtime
            oldest = ts if oldest is None or ts < oldest else oldest

    rows.append(
        (
            tag,
            pcap,
            csv,
            total,
            du_bytes(cell) if cell.is_dir() else 0,
            source,
            ready,
            scored,
        )
    )

print(
    f"{'CELL':20s} {'PCAP':>7s} {'CSV':>8s} {'TOTAL':>8s} {'PCT':>5s} "
    f"{'SIZE':>7s} {'SRC':>4s} {'DS':>3s} {'ATT':>4s}"
)
for tag, pcap, csv, total, size, source, ready, scored in rows:
    pct = total * 100 // TARGET if TARGET else 0
    flag = "" if total or size else "   <- not started"
    print(
        f"{tag:20s} {pcap:7d} {csv:8d} {total:8d} {pct:4d}% {human(size):>7s} "
        f"{source:>4s} {'y' if ready else '-':>3s} {scored:4d}{flag}"
    )

full = TARGET * len(plan)
print(
    f"\ncells     {done_cells}/{len(plan)} collected, "
    f"{sum(1 for r in rows if r[6])} dataset-ready, "
    f"{sum(1 for r in rows if r[7])} scored, {archived} archived"
)
print(
    f"captures  {grand}/{full}  ({grand * 100 // full if full else 0}%)  "
    f"-- {full - grand} remaining"
)
print(f"disk      {human(du_bytes(RESULTS))}")

if oldest and grand:
    hours = (time.time() - oldest) / 3600
    if hours > 0.5:
        rate = grand / hours
        if rate:
            left = (full - grand) / rate
            print(
                f"rate      {rate:.0f} captures/h  -> ~{left:.0f} h "
                f"({left / 24:.1f} days) left"
            )

print(f"\nunprocessed pcaps: {sum(r[1] for r in rows)}")
print(f"in flight (.part): {len(list(RESULTS.glob('*/traces/*.part')))}")
