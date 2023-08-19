import numpy as np
from numba import njit, int32


@njit
def randint(low: int, high: int, num: int) -> np.array:
    if high <= low:
        raise ValueError("high must have a bigger value than low")
    n = high - low
    cdf = get_cdf(n)
    uni_dist = np.random.random(num)
    res = np.empty(num, dtype=int32)
    for i in range(num):
        res[i] = low + bisect(cdf, uni_dist[i])
    return res


@njit
def randint_unique(low: int, high: int, size: int) -> np.array:
    res = np.arange(low, high)
    n = len(res)
    for i in range(min(size, n - 2)):
        j = randint(i, high, 1)[0]
        exchange(res, i, j)
    return res[:size]


@njit
def exchange(arr, i, j):
    if j > i:
        arr[i] += arr[j]
        arr[j] = arr[i] - arr[j]
        arr[i] -= arr[j]


# below are helper functions

@njit
def get_cdf(n: int):
    """
    the probability density function of uniform distribution with bounds (low, high)
    :return:
    """
    res = np.empty(n + 1)
    res[0] = 0.
    res[-1] = 1.
    const = 1. / n
    for i in range(1, n):
        res[i] = const * i
    return res


@njit
def bisect(res, val):
    idx = np.arange(len(res))
    return _bisect(res, val, idx)


@njit
def _bisect(res, val, idx):
    if len(idx) == 1:
        return idx[0]
    tgt = res[idx]
    mid_idx = len(tgt) // 2
    mid_val = tgt[mid_idx]
    if val >= mid_val:
        return _bisect(res, val, idx[mid_idx:])
    else:
        return _bisect(res, val, idx[:mid_idx])

# %timeit randint(2, 5, 100)
# %timeit np.random.randint(0, 2, 100)
# res = np.array([1 / 7 * i for i in range(8)])
# val = 0.15
# bisect(res, val)
