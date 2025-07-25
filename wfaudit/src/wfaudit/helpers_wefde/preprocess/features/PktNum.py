# third party
import numpy as np


# packet number features
def get_packet_counts(times, sizes, multi_conn: bool = True, conn_limit: int = 5):
    sizes = np.asarray(sizes)

    features = []
    if conn_limit <= 0 or not multi_conn:
        return features

    SCALE = 1000
    features = [
        len(sizes) / SCALE,
        len(np.unique(sizes[sizes > 0])) / SCALE,
        len(np.unique(sizes[sizes < 0])) / SCALE,
    ]
    print("PKT", features)

    return features
