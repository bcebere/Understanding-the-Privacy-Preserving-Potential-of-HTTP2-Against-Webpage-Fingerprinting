"""Operating point per (dataset, defense) for the 500-trace main tables.

Selected from the calibration sweep: for each defense on each dataset, the
cheapest level whose attacker F1 is within 0.05 of the best that defense
reaches.  Picking the cheapest rather than the strongest avoids reporting a
saturated cell whose extra cost buys nothing.

    python3 main_table_config.py            # print the plan
    python3 main_table_config.py 2_bbc      # cells for one dataset
"""

import sys

CLIENT = {
    "1_amazon": {
        "front": "mid1",
        "h2pc": "mid2",
        "httpos": "mid2",
        "llama": "high",
        "tamaraw": "mid1",
    },
    "2_bbc": {
        "front": "mid2",
        "h2pc": "high",
        "httpos": "mid2",
        "llama": "high",
        "tamaraw": "lomid",
    },
    "3_reddit": {
        "front": "mid2",
        "h2pc": "high",
        "httpos": "mid2",
        "llama": "high",
        "tamaraw": "high",
    },
    "4_udemy": {
        "front": "mid1",
        "h2pc": "high",
        "httpos": "high",
        "llama": "high",
        "tamaraw": "mid1",
    },
    "5_wiki": {
        "front": "mid2",
        "h2pc": "mid2",
        "httpos": "high",
        "llama": "mid2",
        "tamaraw": "mid2",
    },
}

SERVER = {
    "1_amazon": {"alpaca": "mid1", "tamaraw": "lomid", "h2ps": "high"},
    "2_bbc": {"alpaca": "mid1", "tamaraw": "lomid", "h2ps": "mid2"},
    "3_reddit": {"alpaca": "mid1", "tamaraw": "lomid", "h2ps": "high"},
    "4_udemy": {"alpaca": "mid1", "tamaraw": "lomid", "h2ps": "mid2"},
    "5_wiki": {"alpaca": "mid1", "tamaraw": "lomid", "h2ps": "vhigh"},
}

# Placement dict for server-side defenses
PLACEMENTS = {
    "1_amazon": {
        "1st": "www.amazon.com",
        "3rd_1": "m.media-amazon.com",
        # "3rd_2": "images-na.ssl-images-amazon.com",
        "all": "all",
    },
    "2_bbc": {
        "1st": "www.bbc.com",
        "3rd_1": "static.files.bbci.co.uk",
        # "3rd_2": "ichef.bbci.co.uk",
        "all": "all",
    },
    "3_reddit": {
        "1st": "www.reddit.com",
        "3rd_1": "www.redditstatic.com",
        # "3rd_2": "styles.redditmedia.com",
        "all": "all",
    },
    "4_udemy": {
        "1st": "www.udemy.com",
        "3rd_1": "challenges.cloudflare.com",
        "all": "all",
    },
    "5_wiki": {
        "1st": "en.wikipedia.org",
        "3rd_1": "upload.wikimedia.org",
        # "3rd_2": "login.wikimedia.org",
        "all": "all",
    },
}

FIRST_PARTY = {ds: p["1st"] for ds, p in PLACEMENTS.items()}

# defenses measured at every placement, vs 1st-party only
PER_PLACEMENT = ("alpaca", "tamaraw")

# TODO
CLIENT_ORDER = ["front", "tamaraw", "h2pc", "httpos", "llama"]
SERVER_ORDER = ["alpaca", "tamaraw", "h2ps"]
LEVELS = ["vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh"]


def port(dataset, defense, level):
    """Server port: 9000 + defense*100 + level*10 + dataset id."""
    return (
        9000
        + SERVER_ORDER.index(defense) * 100
        + LEVELS.index(level) * 10
        + int(dataset.split("_")[0])
    )


def cells(dataset):
    """(kind, defense, level, tag, target) for one dataset."""
    out = [("client", "nop", "mid1", "nop", None)]
    for d in CLIENT_ORDER:
        out.append(("client", d, CLIENT[dataset][d], d, None))

    for d in SERVER_ORDER:
        lv = SERVER[dataset][d]
        if d in PER_PLACEMENT:
            for name, target in PLACEMENTS[dataset].items():
                out.append(("server", d, lv, f"srv{d}_{name}", target))
        else:
            out.append(("server", d, lv, f"srv{d}1p", FIRST_PARTY[dataset]))
    return out


if __name__ == "__main__":
    wanted = sys.argv[1:] or sorted(CLIENT)
    total = 0
    for dataset in wanted:
        rows = cells(dataset)
        total += len(rows)
        print(f"\n{dataset}   1st party: {FIRST_PARTY[dataset]}   {len(rows)} cells")
        for kind, defense, level, tag, target in rows:
            if kind == "client":
                print(f"  client  {tag:18s} --defense {defense} --level {level}")
            else:
                print(
                    f"  server  {tag:18s} port {port(dataset, defense, level)}"
                    f"  --level {level}  target {target}"
                )
    print(f"\n{total} cells total")
