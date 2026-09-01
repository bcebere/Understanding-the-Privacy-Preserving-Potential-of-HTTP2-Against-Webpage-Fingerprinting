# third party
MAX_PACKETS = 30


def split_by_value(arr, value):
    result = []
    for idx, num in enumerate(arr):
        if idx == 0 or num == value:
            result.append([])
        result[-1].append(idx)
    result = [s for s in result if s]
    for ridx in result[1:]:  # index 0 need not equal value
        assert (
            arr[ridx[0]] == value
        ), f"Invalid result={ridx} input={arr[ridx[0]]} value={value}"
    return result
