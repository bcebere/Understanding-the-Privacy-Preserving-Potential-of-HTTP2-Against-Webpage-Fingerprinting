# stdlib
from pathlib import Path

# wfaudit absolute
from wfaudit import prepare_all_datasets

workspace = Path("workspace")

prepare_all_datasets(
    workspace=workspace,
    wefde_folder="data",
    prepare_raw_wefde_traces=False,
)
