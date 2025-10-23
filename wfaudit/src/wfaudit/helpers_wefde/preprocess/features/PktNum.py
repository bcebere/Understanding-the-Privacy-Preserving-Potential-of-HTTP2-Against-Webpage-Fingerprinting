# third party
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.preprocess.features.common import split_by_value


# packet number features
def get_packet_counts(times, sizes, conn_limit: int = 1):
    sizes = np.asarray(sizes)

    features = []

    SCALE = 1000

    def _extract_info(pkts):
        if len(pkts) == 0:
            return [0, 0, 0, 0, 0]
        return [
            len(pkts) / SCALE,
            len(pkts[pkts > 0]) / SCALE,
            len(pkts[pkts < 0]) / SCALE,
            len(np.unique(pkts[pkts > 0])) / SCALE,
            len(np.unique(pkts[pkts < 0])) / SCALE,
        ]

    if conn_limit == 1:  # global
        features = _extract_info(sizes)
        # print("PKT global", features)

        return features
    else:
        # per connection
        conn_idxs = split_by_value(times, 0)

        if len(conn_idxs) < conn_limit:
            conn_idxs += [[]] * (conn_limit - len(conn_idxs))

        times = np.asarray(times)
        # Extract conn_limit - 1 connections
        for idx, conn_idx in enumerate(conn_idxs[: conn_limit - 1]):
            features += _extract_info(sizes[conn_idx])

        # Aggregate the rest of the connections
        extra_conns = np.concatenate(conn_idxs[conn_limit - 1 :])
        if len(extra_conns) > 0:
            extra_feats = _extract_info(sizes[extra_conns])
        else:
            extra_feats = _extract_info([])
        features += extra_feats

        assert len(features) == (conn_limit) * 5

        # print(" >>> PKT multi", features)
        return features
