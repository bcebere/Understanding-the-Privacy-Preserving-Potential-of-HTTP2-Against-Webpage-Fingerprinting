def PktLenFeature(times, sizes, features):
    pkt_lens = []
    for i in range(-1500, 1501):
        if i in sizes:
            pkt_lens.append(1)
        else:
            pkt_lens.append(0)
    return pkt_lens
