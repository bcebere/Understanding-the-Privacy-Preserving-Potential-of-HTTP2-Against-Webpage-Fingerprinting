# stdlib
import itertools

# third party
import numpy as np

# wfaudit absolute
from wfaudit.helpers_wefde.preprocess.features.common import split_by_value


def get_cumul_features_per_connection(packets: list, feature_cnt: int):
    if len(packets) == 0:
        packets = [0]
    # Calculate Features

    features = []

    total = []
    pos = []
    neg = []
    inSize = 0
    outSize = 0
    inCount = 0
    outCount = 0

    # Process trace
    for packetsize in itertools.islice(packets, None):
        # CUMUL uses positive to denote incoming, negative to be outgoing,
        # different from dataset
        packetsize = -packetsize

        # incoming packets
        if packetsize >= 0:
            inSize += packetsize
            inCount += 1
            # cumulated packetsizes
            if len(total) == 0:
                total.append(packetsize)
                pos.append(packetsize)
                neg.append(0)
            else:
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + packetsize)
                neg.append(neg[-1] + 0)

        # outgoing packets
        if packetsize <= 0:
            outSize += abs(packetsize)
            outCount += 1
            if len(total) == 0:
                total.append(abs(packetsize))
                pos.append(0)
                neg.append(abs(packetsize))
            else:
                total.append(total[-1] + abs(packetsize))
                pos.append(pos[-1] + 0)
                neg.append(neg[-1] + abs(packetsize))

    # add feature
    features.append(outSize)
    features.append(inSize)

    # cumulative in and out
    posFeatures = np.interp(
        np.linspace(total[0], total[-1], int(feature_cnt / 2)), total, pos
    )
    negFeatures = np.interp(
        np.linspace(total[0], total[-1], int(feature_cnt / 2)), total, neg
    )
    for el in itertools.islice(posFeatures, None):
        features.append(float(el))
    for el in itertools.islice(negFeatures, None):
        features.append(float(el))

    return features


def get_cumul_features(times, packets, feature_cnt=20):
    packets = np.asarray(packets)

    # global
    features = get_cumul_features_per_connection(
        packets.tolist(), feature_cnt=feature_cnt
    )

    print("  >> CUMUL", features)
    return features


def get_cumul_features_multi(
    times, packets, multi_conn: bool = True, feature_cnt=20, conn_limit: int = 10
):
    packets = np.asarray(packets)

    # global
    features = (
        []
    )  # get_cumul_features_per_connection(packets.tolist(), feature_cnt=feature_cnt)

    if conn_limit <= 0 or not multi_conn:
        return features

    # per connection
    conn_idxs = split_by_value(times, 0)

    if len(conn_idxs) < conn_limit:
        conn_idxs += [[]] * (conn_limit - len(conn_idxs))

    for idx, conn_idx in enumerate(conn_idxs[: conn_limit - 1]):
        cumul_raw_stats = get_cumul_features_per_connection(
            packets[conn_idx].tolist(), feature_cnt=feature_cnt
        )
        features += cumul_raw_stats

    # Aggregate the rest of the connections together
    extra_conns = np.concatenate(conn_idxs[conn_limit - 1 :])
    if len(extra_conns) > 0:
        cumul_raw_stats_extra = get_cumul_features_per_connection(
            packets[extra_conns].tolist(), feature_cnt=feature_cnt
        )
    else:
        cumul_raw_stats_extra = get_cumul_features_per_connection(
            [], feature_cnt=feature_cnt
        )
    features += cumul_raw_stats_extra

    assert len(features) == (conn_limit) * (2 + feature_cnt), features

    # print("  >> CUMUL", features)
    return features
