import os

import pandas as pd

DATASETS = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]
DEFENSES = ["front", "h2pc", "httpos", "llama", "tamaraw", "h2srv_1st"] + [
    f"{srvdef}_{srvdep}"
    for srvdef in ["alpaca", "srvtam"]
    for srvdep in ["all", "1st", "3rd_1"]
]

BASE_DIR = "../overhead_analysis/"


def load_csv(path):
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df[df["testcase"] != "testcase"]  # drop duplicate header rows
    for col in ["bytes_rx_real", "bytes_tx_real", "bytes_tx_at_real_end", "latency"]:
        df[col] = pd.to_numeric(df[col])
    return df


def compute_overhead(baseline, defended):
    base = baseline[["testcase", "latency", "bytes_rx_real", "bytes_tx_real"]].rename(
        columns={
            "latency": "latency_base",
            "bytes_rx_real": "bytes_rx_base",
            "bytes_tx_real": "bytes_tx_base",
        }
    )
    deff = defended[
        ["testcase", "latency", "bytes_rx_at_real_end", "bytes_tx_at_real_end"]
    ].rename(
        columns={
            "latency": "latency_def",
            "bytes_rx_at_real_end": "bytes_rx_def",
            "bytes_tx_at_real_end": "bytes_tx_def",
        }
    )
    merged = base.merge(deff, on="testcase")
    merged["delta_down"] = (merged["bytes_rx_def"] - merged["bytes_rx_base"]) / merged[
        "bytes_rx_base"
    ]
    merged["delta_up"] = (merged["bytes_tx_def"] - merged["bytes_tx_base"]) / merged[
        "bytes_tx_base"
    ]
    merged["delta_T"] = (merged["latency_def"] - merged["latency_base"]) / merged[
        "latency_base"
    ]
    return merged[["testcase", "delta_down", "delta_up", "delta_T"]]


# load and pool all datasets
all_overheads = {d: [] for d in DEFENSES}

for dataset in DATASETS:
    baseline_path = os.path.join(BASE_DIR, dataset, "workspace", "ovh_baseline.csv")
    if not os.path.exists(baseline_path):
        print(f"WARNING: missing baseline for {dataset}, skipping")
        continue
    print(dataset)
    baseline = load_csv(baseline_path)
    print(f"{dataset}: {len(baseline)} baseline pages")

    for defense in DEFENSES:
        path = os.path.join(BASE_DIR, dataset, "workspace", f"ovh_{defense}.csv")
        if not os.path.exists(path):
            print(f"WARNING: missing {defense} for {dataset}, skipping")
            continue
        defended = load_csv(path)
        ovh = compute_overhead(baseline, defended)
        all_overheads[defense].append(ovh)

# pool across datasets and report
rows = []
for defense in DEFENSES:
    if not all_overheads[defense]:
        continue
    pooled = pd.concat(all_overheads[defense], ignore_index=True)

    pooled["delta_down"] = pooled["delta_down"].clip(lower=0)
    pooled["delta_up"] = pooled["delta_up"].clip(lower=0)
    pooled["delta_T"] = pooled["delta_T"].clip(lower=0)

    def fmt(median, q1, q3):
        return f"{median:.2f} ({q1:.2f}–{q3:.2f})"

    rows.append(
        {
            "defense": defense,
            "n": len(pooled),
            "∆Up": fmt(
                pooled["delta_up"].median(),
                pooled["delta_up"].quantile(0.25),
                pooled["delta_up"].quantile(0.75),
            ),
            "∆Down": fmt(
                pooled["delta_down"].median(),
                pooled["delta_down"].quantile(0.25),
                pooled["delta_down"].quantile(0.75),
            ),
            "∆T": fmt(
                pooled["delta_T"].median(),
                pooled["delta_T"].quantile(0.25),
                pooled["delta_T"].quantile(0.75),
            ),
        }
    )

summary = pd.DataFrame(rows).set_index("defense")[["n", "∆Up", "∆Down", "∆T"]]
print(summary.to_string())


all_baselines = []
for dataset in DATASETS:
    baseline_path = os.path.join(BASE_DIR, dataset, "workspace", "ovh_baseline.csv")
    if not os.path.exists(baseline_path):
        continue
    baseline = load_csv(baseline_path)
    all_baselines.append(baseline)

baseline_pooled = pd.concat(all_baselines, ignore_index=True)
baseline_check = baseline_pooled[["bytes_tx_real", "bytes_tx_at_real_end"]].describe()
baseline_up_kb = baseline_pooled["bytes_tx_real"].mean() / 1024
baseline_down_kb = baseline_pooled["bytes_rx_real"].mean() / 1024
baseline_T_s = baseline_pooled["latency"].mean()

print("\nBaseline mean:")
print(f"  Upload:   {baseline_up_kb:.2f} KB")
print(f"  Download: {baseline_down_kb:.2f} KB")
print(f"  Latency:  {baseline_T_s:.2f} s")


SRV_DEFENSES = {
    "ALPaCA": {
        "Single Srv.": ["alpaca_1st", "alpaca_3rd_1", "alpaca_3rd_2"],
        "All Srv.": ["alpaca_all"],
    },
    "SRV-TAM": {
        "Single Srv.": ["srvtam_1st", "srvtam_3rd_1", "srvtam_3rd_2"],
        "All Srv.": ["srvtam_all"],
    },
}

print("\n--- SERVER DEFENSE TABLE ---")
for defense_name, deployments in SRV_DEFENSES.items():
    for deploy_label, defense_keys in deployments.items():
        # pool across all keys for this deployment category
        chunks = []
        for defense_key in defense_keys:
            if defense_key not in all_overheads or not all_overheads[defense_key]:
                print(f"WARNING: missing {defense_key}")
                continue
            chunks.append(pd.concat(all_overheads[defense_key], ignore_index=True))

        if not chunks:
            continue

        pooled = pd.concat(chunks, ignore_index=True)
        pooled["delta_down"] = pooled["delta_down"].clip(lower=0)
        pooled["delta_up"] = pooled["delta_up"].clip(lower=0)
        pooled["delta_T"] = pooled["delta_T"].clip(lower=0)

        up = fmt(
            pooled["delta_up"].median(),
            pooled["delta_up"].quantile(0.25),
            pooled["delta_up"].quantile(0.75),
        )
        down = fmt(
            pooled["delta_down"].median(),
            pooled["delta_down"].quantile(0.25),
            pooled["delta_down"].quantile(0.75),
        )
        t = fmt(
            pooled["delta_T"].median(),
            pooled["delta_T"].quantile(0.25),
            pooled["delta_T"].quantile(0.75),
        )
        n = len(pooled)

        print(
            f"  {defense_name:10s} {deploy_label:12s} (n={n:4d})  ∆Up={up}  ∆Down={down}  ∆T={t}"
        )

print(
    f"\nBaseline avg.  ∆Up={baseline_up_kb:.2f} KB  ∆Down={baseline_down_kb:.2f} KB  ∆T={baseline_T_s:.2f} s"
)
