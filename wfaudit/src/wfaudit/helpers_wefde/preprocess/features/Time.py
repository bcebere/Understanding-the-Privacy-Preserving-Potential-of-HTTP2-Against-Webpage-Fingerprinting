# third party
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.preprocess.features.common import split_by_value


# time includes relative timestamps. 0 means a new connection.
def get_time_features_per_connection(times):
    if len(times) == 0:
        return [0, 0, 0]

    return [
        float(np.max(times)),
        float(np.mean(times)),
        float(np.sum(times)),
    ]


def get_time_features(times, sizes):
    # global
    features = get_time_features_per_connection(times)
    return features


def get_time_features_multi(times, sizes, conn_limit: int = 5):
    # global
    features = []  # get_time_features_per_connection(times)

    if conn_limit <= 0:
        return features

    # per connection
    conn_idxs = split_by_value(times, 0)

    if len(conn_idxs) < conn_limit:
        conn_idxs += [[]] * (conn_limit - len(conn_idxs))

    times = np.asarray(times)
    # Extract conn_limit - 1 connections
    for idx, conn_idx in enumerate(conn_idxs[: conn_limit - 1]):
        conn_times = times[conn_idx].tolist()
        conn_times_features = get_time_features_per_connection(conn_times)
        features += conn_times_features

    # Aggregate the rest of the connections
    extra_conns = np.concatenate(conn_idxs[conn_limit - 1 :])
    if len(extra_conns) > 0:
        conn_times_features = get_time_features_per_connection(
            times[extra_conns].tolist()
        )
    else:
        conn_times_features = get_time_features_per_connection([])
    features += conn_times_features

    assert len(features) == (conn_limit) * 3

    # print(" >>> TIME", features)
    return features
