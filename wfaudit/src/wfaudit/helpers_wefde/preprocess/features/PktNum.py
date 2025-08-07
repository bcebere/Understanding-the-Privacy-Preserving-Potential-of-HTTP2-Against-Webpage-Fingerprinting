# third party
import numpy as np


# packet number features
def get_packet_counts(times, sizes):
    sizes = np.asarray(sizes)

    features = []

    SCALE = 1000
    features = [
        len(sizes) / SCALE,
        len(sizes[sizes > 0]) / SCALE,
        len(sizes[sizes < 0]) / SCALE,
        len(np.unique(sizes[sizes > 0])) / SCALE,
        len(np.unique(sizes[sizes < 0])) / SCALE,
    ]
    # print("PKT", features)

    return features
