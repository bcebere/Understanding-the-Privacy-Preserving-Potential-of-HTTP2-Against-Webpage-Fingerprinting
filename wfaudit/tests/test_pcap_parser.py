# stdlib
from pathlib import Path
import shutil

# third party
import pytest

# wfaudit absolute
from wfaudit import (
    _prepare_time_series_arrow,
    merge_pcap_csvs,
    prepare_all_datasets,
    prepare_deepse_dataset,
    prepare_wefde_dataset,
    prepare_wefde_raw,
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
    files = process_raw_pcaps(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    static_ds, temporal_ds = merge_pcap_csvs(workspace=tmp_path)

    static_proc, ts_proc, (lens, sizes, rel_times) = _prepare_time_series_arrow(
        static_ds, temporal_ds
    )

    assert len(static_proc) > 0
    assert len(static_proc) == len(files)
    assert len(ts_proc) == len(static_proc)
    assert len(lens) == len(files)


def test_prepare_wefde_datasets(tmp_path):
    pcaps = process_raw_pcaps(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    static_data, ts_data = merge_pcap_csvs(workspace=tmp_path)

    print("Creating wefde raw traces")
    wefde_folder = "wefde_dbg"
    prepare_wefde_raw(
        static_data, ts_data, workspace=tmp_path, wefde_folder=wefde_folder
    )

    output = tmp_path / wefde_folder
    assert output.exists()

    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == len(pcaps)

    print("Creating wefde features")
    wefde_feats_folder = "wefde_feats_dbg"
    prepare_wefde_dataset(
        workspace=tmp_path,
        wefde_folder=wefde_folder,
        wefde_feats_folder=wefde_feats_folder,
    )
    output = tmp_path / wefde_feats_folder
    assert output.exists()
    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == len(pcaps) + 1


@pytest.mark.parametrize("testtype", ["real", "sanity"])
def test_prepare_deepse_datasets(tmp_path, testtype):
    process_raw_pcaps(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    static_data, ts_data = merge_pcap_csvs(workspace=tmp_path)

    print("Creating wefde raw traces")
    wefde_folder = "wefde_dbg"
    prepare_wefde_raw(
        static_data, ts_data, workspace=tmp_path, wefde_folder=wefde_folder
    )

    path_wefde = tmp_path / wefde_folder
    assert path_wefde.exists()

    output = tmp_path / "output_deepse" / testtype / "dataset.npz"

    print("Creating deepse features")
    prepare_deepse_dataset(
        path_wefde=path_wefde,
        path_out=output,
        n_websites=3,
        n_traces=2,
        debug_mode=(testtype != "real"),
    )
    assert output.exists()


def test_e2e(tmp_path):
    pcaps = process_raw_pcaps(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    deepse_testtypes = ["real", "sanity111"]
    wefde_folder = "wefde_dbg"
    wefde_feats_folder = "wefde_feats_dbg"
    prepare_all_datasets(
        workspace=tmp_path,
        n_websites=3,
        n_traces=2,
        deepse_testtypes=deepse_testtypes,
        wefde_folder=wefde_folder,
        wefde_feats_folder=wefde_feats_folder,
    )

    output = tmp_path / "output_csv_single"
    assert output.exists()
    is_empty = not any(output.iterdir())
    assert not is_empty

    output = tmp_path / wefde_folder
    assert output.exists()
    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == len(pcaps)

    output = tmp_path / wefde_feats_folder
    assert output.exists()
    num_files = sum(1 for file in output.iterdir() if file.is_file())
    assert num_files == len(pcaps) + 1

    for testtype in deepse_testtypes:
        output = tmp_path / "output_deepse" / testtype / "dataset.npz"
        assert output.exists()
