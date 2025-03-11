# stdlib
from pathlib import Path
import shutil

# third party
import numpy as np
import pandas as pd
import pytest

# wfaudit absolute
from wfaudit import (
    create_datasets,
    merge_pcap_csvs,
    prepare_features,
    prepare_ts_datasets,
    prepare_ts_datasets_for_nns_1C,
    prepare_ts_datasets_for_nns_3C,
    process_raw_pcaps,
)


def test_process_raw_pcaps_sanity():
    process_raw_pcaps(Path("fake_path"))


def test_process_raw_pcaps_correctness(tmp_path):
    traces = Path("traces")
    process_raw_pcaps(
        traces=traces,
        workspace=Path(tmp_path),
        unlink_after_processing=False,
    )
    output = tmp_path / "output_csv_single"
    assert output.exists()
    for trace in traces.iterdir():
        stem = trace.stem

        assert (output / f"static_data_{stem}.csv").exists()
        assert (output / f"temporal_data_{stem}.csv").exists()


@pytest.fixture
def setup_and_teardown():
    # Setup code
    pass

    yield

    # Teardown code
    traces = Path("traces_drop")

    if traces.exists():
        shutil.rmtree(traces)


def test_process_raw_pcaps_unlink(tmp_path, setup_and_teardown):
    orig_traces = Path("traces")
    traces = Path("traces_drop")

    if traces.exists():
        shutil.rmtree(traces)
    shutil.copytree(orig_traces, traces)

    num_files = sum(1 for file in traces.iterdir() if file.is_file())
    assert num_files == 6

    process_raw_pcaps(
        traces=traces,
        workspace=Path(tmp_path),
        unlink_after_processing=True,
    )

    num_files = sum(1 for file in traces.iterdir() if file.is_file())
    assert num_files == 0


def test_merge_csv_pcaps(tmp_path):
    process_raw_pcaps(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    merge_pcap_csvs(workspace=tmp_path)
    output = tmp_path / "output_csv_full"

    assert output.exists()
    assert (output / "static_data.csv").exists()
    assert (output / "temporal_data.csv").exists()

    static_data = pd.read_csv(output / "static_data.csv")
    assert len(static_data) == 6
    assert "file_order" in static_data.columns

    temporal_data = pd.read_csv(output / "temporal_data.csv")
    assert "file_order" in temporal_data.columns
    assert len(temporal_data["file_order"].unique()) == 6


def test_prepare_wefde_datasets(tmp_path):
    process_raw_pcaps(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    merge_pcap_csvs(workspace=tmp_path)

    prepare_ts_datasets(workspace=tmp_path)

    output = tmp_path / "output_wefde"
    assert output.exists()

    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == 6


def test_prepare_nn_datasets_simple(tmp_path):
    process_raw_pcaps(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    merge_pcap_csvs(workspace=tmp_path)

    prepare_ts_datasets(workspace=tmp_path)
    prepare_features(workspace=tmp_path, conn_limit=1)
    prepare_ts_datasets_for_nns_1C(workspace=tmp_path)

    output = tmp_path / "output_ml"
    assert output.exists()

    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == 2

    with open(output / "X_1C.npy", "rb") as f:
        X = np.load(f)
    with open(output / "y_1C.npy", "rb") as f:
        y = np.load(f)

    assert len(X) == len(y)
    assert X.shape[1] == 1  # size, ts
    assert X.shape[2] == 19


def test_prepare_nn_datasets(tmp_path):
    process_raw_pcaps(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    merge_pcap_csvs(workspace=tmp_path)

    prepare_ts_datasets_for_nns_3C(workspace=tmp_path)

    output = tmp_path / "output_ml"
    assert output.exists()

    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == 2

    with open(output / "X_3C.npy", "rb") as f:
        X = np.load(f)
    with open(output / "y_3C.npy", "rb") as f:
        y = np.load(f)

    assert len(X) == len(y)
    assert X.shape[1] == 3  # size, ts
    assert X.shape[2] == 225


def test_e2e(tmp_path):
    create_datasets(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    output = tmp_path / "output_csv_single"
    assert output.exists()

    output = tmp_path / "output_csv_full"
    assert output.exists()

    output = tmp_path / "output_wefde"
    assert output.exists()

    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == 6
