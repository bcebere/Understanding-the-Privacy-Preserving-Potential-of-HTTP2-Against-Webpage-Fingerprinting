# third party
import dill
import numpy as np
import pandas as pd

# feature_range = [
#    ("Time Statistics", (1, 3)),
#    ("Burst Statistics", (4, 6)),
#    ("CUMUL", (7, 16)),
# ]


def print_leakage(
    feature_range: dict,  # offsets for leakage types
    dataset_name: str,  # dataset,
    indiv_file: str,  # path to individual leakages,
    joint_file: str,  # path to joint leakages,
    top_joint_file: str,  # top leakages,
    output: str,  # path to write to summary,
):
    # read independent leakage information from files
    with open(indiv_file, "rb") as fi:
        leakages = dill.load(fi)
    with open(joint_file, "rb") as fi:
        joint_leakages = dill.load(fi)
    with open(top_joint_file, "rb") as fi:
        top_joint_leakages = dill.load(fi)

    assert len(leakages) == feature_range[-1][1][1]

    summary = {"testcase": dataset_name}
    for i in range(1, len(feature_range) + 1):
        category, indices = feature_range[i - 1]

        # plot category leakages for each leakage file
        # x = range(indices[0], indices[1] + 1)
        y = leakages[indices[0] - 1 : indices[1]]

        for dbg in range(len(y)):
            if y[dbg] > 7:
                print(category, dbg, y[dbg], len(y))

        summary[category] = [np.max(y)]

    summary["joint"] = joint_leakages[0]
    summary["top_joint"] = top_joint_leakages[0]

    summary = pd.DataFrame(summary)
    summary.to_csv(output, index=None)
    print(output, summary)
