# stdlib
from pathlib import Path

# wfaudit absolute
from wfaudit import prepare_all_datasets, process_raw_pcaps

workspace = Path("workspace")
pcaps = process_raw_pcaps(
    traces=workspace / "traces",
    workspace=workspace,
    unlink_after_processing=False,
)

prepare_all_datasets(
    workspace=workspace,
    n_websites=3,
    n_traces=150,
)
