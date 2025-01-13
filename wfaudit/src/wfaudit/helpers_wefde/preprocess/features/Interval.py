# inflow interval (icics, knn)
# third party
from features.common import MAX_PACKETS, X


def IntervalFeature(times, sizes, Category):
    local_features = []
    if Category == "KNN":
        # a list of first MAX_PACKETS intervals (KNN)
        # incoming interval
        count = 0
        prevloc = 0
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                count += 1
                local_features.append(i - prevloc)
                prevloc = i
            if count == MAX_PACKETS:
                break
        for i in range(count, MAX_PACKETS):
            local_features.append(X)

        # outgoing interval
        count = 0
        prevloc = 0
        for i in range(0, len(sizes)):
            if sizes[i] < 0:
                count += 1
                local_features.append(i - prevloc)
                prevloc = i
            if count == MAX_PACKETS:
                break
        for i in range(count, MAX_PACKETS):
            local_features.append(X)

    if Category == "ICICS" or Category == "WPES11":
        MAX_INTERVAL = MAX_PACKETS
        # Distribution of the intervals
        # incoming interval
        count = 0
        prevloc = 0
        interval_freq_in = [0] * (MAX_INTERVAL + 1)
        for i in range(0, len(sizes)):
            if sizes[i] > 0:
                inv = i - prevloc - 1
                prevloc = i
                # record the interval
                if inv > MAX_INTERVAL:
                    inv = MAX_INTERVAL
                interval_freq_in[inv] += 1

        # outgoing interval
        count = 0
        prevloc = 0
        interval_freq_out = [0] * (MAX_INTERVAL + 1)
        for i in range(0, len(sizes)):
            if sizes[i] < 0:
                inv = i - prevloc - 1
                prevloc = i
                # record the interval
                if inv > MAX_INTERVAL:
                    inv = MAX_INTERVAL
                interval_freq_out[inv] += 1

        # ICICS: no grouping
        if Category == "ICICS":
            local_features.extend(interval_freq_in)
            local_features.extend(interval_freq_out)

        # WPES 11: 1, 2, 3-5, 6-8, 9-13, 14 (grouping)
        if Category == "WPES11":
            # incoming
            local_features.extend(interval_freq_in[0:3])
            local_features.append(sum(interval_freq_in[3:6]))
            local_features.append(sum(interval_freq_in[6:9]))
            local_features.append(sum(interval_freq_in[9:14]))
            local_features.extend(interval_freq_in[14:])
            # outgoing
            local_features.extend(interval_freq_out[0:3])
            local_features.append(sum(interval_freq_out[3:6]))
            local_features.append(sum(interval_freq_out[6:9]))
            local_features.append(sum(interval_freq_out[9:14]))
            local_features.extend(interval_freq_out[14:])

    return local_features
