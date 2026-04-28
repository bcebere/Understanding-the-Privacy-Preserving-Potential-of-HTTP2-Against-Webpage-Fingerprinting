# stdlib
from pathlib import Path

# wfaudit absolute
from wfaudit import prepare_all_datasets

workspace = Path("workspace")
# Parse PCAPs, if needed
# pcaps = process_raw_pcaps(
#    traces=workspace / "traces",
#    workspace=workspace,
#    unlink_after_processing=False,
# )

prepare_all_datasets(
    workspace=workspace,
    wefde_folder="data",
)
