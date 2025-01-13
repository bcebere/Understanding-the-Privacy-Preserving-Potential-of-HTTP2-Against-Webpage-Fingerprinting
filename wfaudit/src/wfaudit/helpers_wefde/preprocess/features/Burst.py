# third party
import numpy as np


# knn feature (share similarity with interval)
# the burst of inflow traffic
def BurstFeature(times, sizes, max_length=20):
    burst_features = []
    bursts = []
    for x in sizes:
        if x > 0:
            bursts.append(x)
        else:
            bursts.append(-x)

    # burst could be none
    if len(bursts) != 0:
        burst_features = [
            np.max(bursts),
            np.mean(bursts),
            np.std(bursts),
        ]
    else:
        burst_features = [0, 0, 0, 0, 0]

    return burst_features
