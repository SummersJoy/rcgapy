import numpy as np
from numba import njit


@njit
def bisect(res, val):
    if val < res[0]:
        return -1
    idx = np.arange(len(res))
    return _bisect(res, val, idx)


@njit(fastmath=True)
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
