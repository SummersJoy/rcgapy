import numpy as np
from numba import njit, int32


def get_dist_mat(cx: np.array, cy: np.array, decimals: int = 3) -> np.array:
    if len(cx) != len(cy):
        raise ValueError("x y have different length")
    n = len(cx)
    res = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            res[i, j] = np.round(np.sqrt((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2), decimals)
    return res


@njit
def fill_zero(n: int, arr: np.array) -> np.array:
    tmp = np.zeros(n + 1, dtype=int32)
    tmp[1:] = arr
    return tmp


def reformat_depot(coord: np.array) -> np.array:
    n = len(coord)
    tmp = np.empty(n)
    tmp[1:] = coord[:n - 1]
    tmp[0] = coord[n - 1]
    return tmp
