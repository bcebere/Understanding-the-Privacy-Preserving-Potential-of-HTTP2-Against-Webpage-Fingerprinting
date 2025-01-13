# stdlib
from pathlib import Path
import shutil

# third party
import pandas as pd
import pytest

# wfaudit absolute
from wfaudit import (
    create_datasets,
    merge_pcap_csvs,
    prepare_wefde_datasets,
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

    prepare_wefde_datasets(workspace=tmp_path)

    output = tmp_path / "output_wefde"
    assert output.exists()

    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == 6


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
