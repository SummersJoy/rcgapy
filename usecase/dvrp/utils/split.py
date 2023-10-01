import numpy as np
from numba import njit, int32


@njit(fastmath=True)
def split(n: int, s: np.ndarray, q: np.ndarray, d: np.ndarray, c: np.ndarray, w: float, max_load: float) \
        -> tuple[np.ndarray, float]:
    """
    Bellman algorithm to split a permutation of route into several feasible sub-routes with minimum travel distance
    """
    v = np.empty(n + 1)
    p = np.empty(n + 1, dtype=int32)
    v[0] = 0.
    v[1:] = 999999.
    for i in range(1, n + 1):
        load = 0.
        cost = 0.
        j = i
        while True:
            sj = s[j]
            sj_prev = s[j - 1]
            load += q[sj]
            if i == j:
                cost = c[0, sj] + d[sj] + c[sj, 0]
            else:
                cost = cost - c[sj_prev, 0] + c[sj_prev, sj] + d[sj] + c[sj, 0]
            if load <= w and cost <= max_load:
                if v[i - 1] + cost < v[j]:
                    v[j] = v[i - 1] + cost
                    p[j] = i - 1
                j += 1
            if j >= n or load > w or cost > max_load:
                break
    return p, v[-1]


@njit
def label2route(n, p, s, max_rl):
    trip = np.zeros((n, max_rl + 1), dtype=int32)
    t = -1
    j = n
    while True:
        t += 1
        i = p[j]
        count = 0
        for k in range(i + 1, j + 1):
            trip[t, count] = s[k]
            count += 1
        j = i
        if i == 0:
            break
    return trip[:(t + 1), :]


@njit
def get_max_route_len(q: np.ndarray, w: float) -> int:
    """
    Compute the max number of possible customer in each trip
    """
    n = len(q) - 1
    demand = sorted(q[1:])
    current = 0
    for i in range(n):
        current += demand[i]
        if current > w:
            return i - 1
    return n

# p, fitness = split(n, s, q, d, c, w, max_load)
# trip = route_retrieve(n, p, s)
