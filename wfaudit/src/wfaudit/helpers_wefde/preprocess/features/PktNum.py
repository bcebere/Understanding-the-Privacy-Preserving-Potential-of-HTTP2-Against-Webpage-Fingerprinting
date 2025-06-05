# third party
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.preprocess.features.common import split_by_value


# packet number features
def get_packet_counts(times, sizes, multi_conn: bool = True, conn_limit: int = 5):
    sizes = np.asarray(sizes)

    features = []
    if conn_limit <= 0 or not multi_conn:
        return features

    # per connection
    conn_idxs = split_by_value(times, 0)
    if len(conn_idxs) < conn_limit:
        conn_idxs += [[]] * (conn_limit - len(conn_idxs))

    for idx, conn_idx in enumerate(conn_idxs[:conn_limit]):
        conn_packets = sizes[conn_idx]
        features.extend(
            [
                len(conn_idx),
                len(np.unique(conn_packets[conn_packets > 0])),
                len(np.unique(conn_packets[conn_packets < 0])),
            ]
        )
    assert len(features) == 3 * conn_limit, features

    return features
