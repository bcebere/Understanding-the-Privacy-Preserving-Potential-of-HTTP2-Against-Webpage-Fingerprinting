# third party
import numpy

PACKET_MOD = 3
MAX_PACKETS = 20


def PktDistFeature(times, sizes):
    count = 0
    temp = []
    local_features = []
    for i in range(0, min(len(sizes), 6000)):
        if sizes[i] > 0:
            count += 1
        if i % PACKET_MOD == PACKET_MOD - 1:
            local_features.append(count)
            temp.append(count)
            count = 0
    for i in range(len(sizes) // PACKET_MOD, MAX_PACKETS):
        local_features.append(0)
        temp.append(0)
    # std
    local_features.append(numpy.std(temp))
    # mean
    local_features.append(numpy.mean(temp))
    # median
    local_features.append(numpy.median(temp))
    # max
    local_features.append(numpy.max(temp))

    # alternative packet distribution list (k-anonymity)
    # could be considered packet distributions with larger intervals
    num_bucket = PACKET_MOD
    bucket = [0] * (num_bucket + 1)
    for i in range(0, MAX_PACKETS):
        ib = i // (MAX_PACKETS // num_bucket)
        bucket[ib] = bucket[ib] + temp[i]
    local_features.extend(bucket)
    local_features.append(numpy.sum(bucket))

    return local_features
