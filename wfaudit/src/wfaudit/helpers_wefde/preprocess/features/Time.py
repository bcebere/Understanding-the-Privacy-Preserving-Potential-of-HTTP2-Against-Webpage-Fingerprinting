# third party
from features.common import X
import numpy


# max, mean, std, quartile
def interTimeStats(times):
    res = []
    for i in range(1, len(times)):
        prev = times[i - 1]
        cur = times[i]
        res.append(cur - prev)

    if len(res) == 0:
        return [X, X, X, X]
    else:
        return [
            numpy.max(res),
            numpy.mean(res),
            numpy.std(res),
        ]


# k-anonymity
# inter packet time statistics for total, incoming, and outgoing
# max, mean, std, third quartile
def TimeFeature(times, sizes):
    features = []
    # inter packet time feature
    # total
    features.extend(interTimeStats(times))
    # outgoing
    # times_out = []
    # for i in range(0, len(sizes)):
    #    if sizes[i] >= 0:
    #        times_out.append(times[i])
    # features.extend(interTimeStats(times_out))

    # incoming
    # times_in = []
    # for i in range(0, len(sizes)):
    #    if sizes[i] <= 0:
    #        times_in.append(times[i])
    # features.extend(interTimeStats(times_in))

    return features
