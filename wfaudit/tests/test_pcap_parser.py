# stdlib
from pathlib import Path
import shutil

# third party
import pytest

# wfaudit absolute
from wfaudit import process_raw_pcaps


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
