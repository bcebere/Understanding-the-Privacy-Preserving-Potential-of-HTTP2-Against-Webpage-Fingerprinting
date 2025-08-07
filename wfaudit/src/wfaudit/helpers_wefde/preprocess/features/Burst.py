# stdlib
import heapq

# third party
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.preprocess.features.common import split_by_value


def get_burst_features_per_connection(bursts: list, topn: int):
    if len(bursts) < topn:
        bursts += [0] * (topn - len(bursts))

    burst_features = heapq.nlargest(topn, bursts)
    burst_features += [
        float(np.mean(bursts)),
        float(np.std(bursts)),
        float(np.sum(bursts)),
    ]
    return burst_features


def get_burst_features(times, sizes, topn: int = 20):
    sizes = np.abs(np.asarray(sizes))
    burst_features = get_burst_features_per_connection(
        np.unique(sizes).astype(float).tolist(), topn=topn
    )
    # print("BURST", burst_features)

    return burst_features


# times are relative to previous packet. 0 means a new connection
def get_burst_featuresi_multi(
    times, sizes, multi_conn: bool = True, topn: int = 20, conn_limit: int = 5
):
    sizes = np.abs(np.asarray(sizes))

    # global
    burst_features = []  # get_burst_features_per_connection(sizes.tolist(), topn=topn)

    if conn_limit <= 0 or not multi_conn:
        return burst_features

    # per connection
    conn_idxs = split_by_value(times, 0)

    if len(conn_idxs) < conn_limit:
        conn_idxs += [[]] * (conn_limit - len(conn_idxs))

    def _process_connection(conn_idx):
        if len(conn_idx) == 0:
            conn_bursts = []
        else:
            conn_bursts = sizes[conn_idx].tolist()
        conn_burst_features = get_burst_features_per_connection(conn_bursts, topn=topn)
        conn_burst_uniq_features = get_burst_features_per_connection(
            np.unique(conn_bursts).astype(float).tolist(), topn=topn
        )

        return conn_burst_features, conn_burst_uniq_features

    # Extract first conn_limit - 1 connections
    for idx, conn_idx in enumerate(conn_idxs[: conn_limit - 1]):
        print("Eval single conn", conn_idx)
        _, conn_burst_uniq_features = _process_connection(conn_idx)
        burst_features += conn_burst_uniq_features
    # Aggregate the rest of the connections together
    extra_conns = np.concatenate(conn_idxs[conn_limit - 1 :])
    _, conn_burst_uniq_features = _process_connection(extra_conns)
    burst_features += conn_burst_uniq_features

    assert len(burst_features) == (conn_limit) * (topn + 3), burst_features

    # print(" >>> BURST", burst_features)

    return burst_features
