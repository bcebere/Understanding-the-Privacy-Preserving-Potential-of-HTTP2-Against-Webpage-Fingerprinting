# wfaudit absolute
from wfaudit.helpers_wefde.preprocess.features.common import split_by_value


# packet number features
def get_packet_counts(times, sizes, multi_conn: bool = True, conn_limit: int = 5):
    features = []
    if conn_limit <= 0 or not multi_conn:
        return features

    # per connection
    conn_idxs = split_by_value(times, 0)
    if len(conn_idxs) < conn_limit:
        conn_idxs += [[]] * (conn_limit - len(conn_idxs))

    for idx, conn_idx in enumerate(conn_idxs[:conn_limit]):
        features.append(len(conn_idx))

    assert len(features) == conn_limit, features

    return features
