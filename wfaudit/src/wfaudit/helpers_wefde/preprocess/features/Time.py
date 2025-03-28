# third party
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.preprocess.features.common import split_by_value


# time includes relative timestamps. 0 means a new connection.
def get_time_features_per_connection(times):
    if len(times) == 0:
        return [0, 0, 0, 0]

    return [
        float(np.max(times)),
        float(np.mean(times)),
        float(np.std(times)),
        float(np.sum(times)),
    ]


def get_time_features(times, sizes, multi_conn: int = True, conn_limit: int = 5):
    # global
    features = []  # get_time_features_per_connection(times)

    if conn_limit <= 0 or not multi_conn:
        return features

    # per connection
    conn_idxs = split_by_value(times, 0)

    if len(conn_idxs) < conn_limit:
        conn_idxs += [[]] * (conn_limit - len(conn_idxs))

    times = np.asarray(times)
    for idx, conn_idx in enumerate(conn_idxs[:conn_limit]):
        conn_times = times[conn_idx].tolist()
        conn_times_features = get_time_features_per_connection(conn_times)

        features += conn_times_features

    assert len(features) == (conn_limit) * 4

    return features
