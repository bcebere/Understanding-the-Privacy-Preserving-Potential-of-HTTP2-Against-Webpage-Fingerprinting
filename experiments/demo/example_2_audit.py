# Evaluate the security of the dataset.
# NOTE: this is just a toy dataset. For the neural network estimators, we need at least 500 samples per class, for decent performance.

# stdlib
from pathlib import Path

# wfaudit absolute
from wfaudit import audit

workspace = Path("workspace")
ml_output_folder = workspace / "eval_ml"
wefde_output_folder = workspace / "eval_wefde"
deepse_output = workspace / "eval_deepse/results.csv"
xai_output_folder = workspace / "eval_xai"

wefde_feats_folder = workspace / "output_features"
deepse_dataset = workspace / "output_deepse" / "real" / "dataset.npz"

scores = audit(
    # ML
    ml_output_folder=ml_output_folder,
    wefde_feats_folder=wefde_feats_folder,
    deepse_dataset=deepse_dataset,
    ml_arch_2D=["xgboost"],
    ml_arch_3D=[],
    # leakage
    wefde_output_folder=wefde_output_folder,
    deepse_output=deepse_output,
    # xai
    xai_output_folder=xai_output_folder,
)

print("ML scores ---> ", scores["ML"])
print("Leakage scores ---> ", scores["leakage"])
print("XAI scores ---> ", scores["xai"])
