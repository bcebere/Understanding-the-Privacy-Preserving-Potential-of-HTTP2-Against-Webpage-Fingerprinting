# stdlib
import heapq

# third party
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.preprocess.features.common import split_by_value


def get_burst_features_per_connection(bursts, topn=10):
    if len(bursts) < topn:
        bursts += [0] * (topn - len(bursts))

    burst_features = heapq.nlargest(topn, bursts)
    burst_features += [
        float(np.mean(bursts)),
        float(np.median(bursts)),
        float(np.std(bursts)),
        float(np.sum(bursts)),
    ]
    # burst_features.extend(bursts[ : topn])
    return burst_features


# times are relative to previous packet. 0 means a new connection
def get_burst_features(
    times, sizes, multi_conn: bool = True, topn: int = 10, conn_limit: int = 5
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

    for idx, conn_idx in enumerate(conn_idxs[:conn_limit]):
        conn_bursts = sizes[conn_idx].tolist()
        conn_burst_features = get_burst_features_per_connection(conn_bursts, topn=topn)
        conn_burst_uniq_features = get_burst_features_per_connection(
            np.unique(conn_bursts).astype(float).tolist(), topn=topn
        )

        burst_features += conn_burst_features
        burst_features += conn_burst_uniq_features

    assert len(burst_features) == 2 * (conn_limit) * (topn + 4), burst_features
    return burst_features
