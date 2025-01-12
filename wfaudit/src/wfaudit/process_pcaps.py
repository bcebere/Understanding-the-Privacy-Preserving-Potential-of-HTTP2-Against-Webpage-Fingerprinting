# stdlib
import glob
import time
from pathlib import Path

# third party
from tqdm import tqdm

# wfaudit absolute
from wfaudit.processing import process_pcap


def process_pcaps(
    traces=Path("traces"),
    workspace=Path("workspace"),
    unlink_after_processing=True,
):
    """
    Args:
        - traces: Folder with the PCAPs to be parsed. --- traces / "*.pcap"
        - workspace : Folder where to store intermediary and final CSVs.
        - unlink_after_processing: Delete the PCAP after processing. Useful for low-space devices.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / "output_csv_single"
    output.mkdir(parents=True, exist_ok=True)

    files = glob.glob(str(traces / "*.pcap"))

    for filename in tqdm(files):
        filename = Path(filename)
        stem = filename.stem
        output_csv_static = output / f"static_data_{stem}.csv"
        output_csv_temporal = output / f"temporal_data_{stem}.csv"
        if not filename.exists():
            continue

        if output_csv_temporal.exists():
            print("already cached", output_csv_temporal)
            if unlink_after_processing:
                print("dropping ", filename)
                filename.unlink()
            continue
        try:
            session = process_pcap(filename)
        except BaseException as e:
            print("failed to parse pcap. moving to graveyard", filename, e)
            filename.unlink()
            time.sleep(0.1)
            continue

        label = stem.split("_")[1]
        static_data, temporal_data = session.temporal_stats_per_flow()
        if len(static_data) == 0:
            print("empty dataset", filename)
            filename.unlink()
            continue

        static_data["label"] = label
        # print(filename, len(static_data), len(temporal_data))
        static_data.to_csv(output_csv_static, index=False)
        temporal_data.to_csv(output_csv_temporal, index=False)

        if unlink_after_processing:
            print("dropping ", filename)
            filename.unlink()
