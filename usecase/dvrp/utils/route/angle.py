import numpy as np
from numba import njit, int32


@njit
def get_angle(cx, cy):
    """
    Angle of each customer against the depot
    """
    n = len(cx)
    res = np.empty(n)
    for i in range(1, n):
        dx = cx[i] - cx[0]
        dy = cy[i] - cy[0]
        if dx == 0.:
            if dy >= 0.:
                res[i] = 0.
            else:
                res[i] = 180.0
        elif dx > 0.:
            if dy == 0.:
                res[i] = 90.0
            elif dy > 0.:
                res[i] = np.arctan(dy / dx) * 180.0 / np.pi
            else:
                res[i] = 360.0 - np.arctan(-dy / dx) * 180.0 / np.pi
        elif dy == 0.:
            res[i] = 270.0
        elif dy > 0:
            res[i] = 180.0 - np.arctan(-dy / dx) * 180.0 / np.pi
        else:
            res[i] = 180.0 + np.arctan(dy / dx) * 180.0 / np.pi
    return np.round(res, 3)


@njit
def near_neighbor(angle, threshold=45):
    n = len(angle)
    res = np.zeros((n, n), dtype=int32)
    max_count = 0
    for i in range(1, n):
        agl = angle[i]
        count = 0
        for j in range(1, n):
            if i != j and agl - threshold <= angle[j] <= agl + threshold:
                res[i, count] = j
                count += 1
        if count > max_count:
            max_count = count
    return res[:, :max_count]
