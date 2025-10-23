# third party
MAX_PACKETS = 30


# times are relative to previous packet. 0 means a separate connection
def split_by_value(arr, value):
    result = [[]]

    for idx, num in enumerate(arr):
        if (
            num == value
        ):  # When the split value is encountered, push to result and reset
            result.append([])

        result[-1].append(idx)

    result = [sublist for sublist in result if len(sublist) > 0]
    for ridx in result:
        assert arr[ridx[0]] == value, f"Invalid result={ridx} input={arr[ridx[0]]}"

    return result
