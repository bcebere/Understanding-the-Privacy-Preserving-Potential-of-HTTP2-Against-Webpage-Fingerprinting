# extract params
FEATURE_EXT = ".features"

# packet number per second, how many milliseconds to count?
howlong = 1000

# n-gram feature
NGRAM = 3

# CUMUL feature number
featureCount = 8


# Python3 conversion of python2 cmp function
def cmp(a, b):
    return (a > b) - (a < b)


# normalize traffic
def normalize_traffic(times, sizes):
    # sort
    tmp = sorted(zip(times, sizes))

    times = [x for x, _ in tmp]
    sizes = [x for _, x in tmp]

    TimeStart = times[0]
    PktSize = 500

    # normalize time
    for i in range(len(times)):
        if times[i] == 0:
            raise
        times[i] = times[i] - TimeStart

    # normalize size
    for i in range(len(sizes)):
        sizes[i] = (abs(sizes[i]) / PktSize) * cmp(sizes[i], 0)

    # flat it
    newtimes = list()
    newsizes = list()

    for t, s in zip(times, sizes):
        numCell = abs(s)
        oneCell = cmp(s, 0)
        for r in range(numCell):
            newtimes.append(t)
            newsizes.append(oneCell)

    return newtimes, newsizes
