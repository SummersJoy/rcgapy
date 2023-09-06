import numpy as np


def get_dist_mat(cx, cy, decimals=3):
    if len(cx) != len(cy):
        raise ValueError("x y have different length")
    n = len(cx)
    res = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            res[i, j] = np.round(np.sqrt((cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2), decimals)
    return res


def fill_zero(n, arr):
    tmp = np.zeros(n + 1)
    tmp[1:] = arr
    return tmp


def reformat_depot(coord):
    n = len(coord)
    tmp = np.empty(n)
    tmp[1:] = coord[:n-1]
    tmp[0] = coord[n - 1]
    return tmp