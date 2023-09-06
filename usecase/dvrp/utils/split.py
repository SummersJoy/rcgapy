import numpy as np
from numba import njit, int32


# n = 10
# perm = np.random.permutation(10) + 1
# s = np.zeros(n + 1, dtype=int)
# s[0] = 0
# s[1:] = perm
# q = np.random.randint(1, 10, 11)
# q[0] = 0
# d = np.zeros(11)
# d[0] = 0
# c = np.random.random((11, 11))
# w = 20
# max_load = 100


@njit
def split(n, s, q, d, c, w, max_load):
    v = np.empty(n + 1)
    p = np.empty(n + 1, dtype=int32)
    v[0] = 0
    v[1:] = np.inf
    for i in range(1, n + 1):
        load = 0
        cost = 0
        j = i
        while True:
            load += q[s[j]]
            if i == j:
                cost = c[0, s[j]] + d[s[j]] + c[s[j], 0]
            else:
                cost = cost - c[s[j - 1], 0] + c[s[j - 1], s[j]] + d[s[j]] + c[s[j], 0]
            if load <= w and cost <= max_load:
                if v[i - 1] + cost < v[j]:
                    v[j] = v[i - 1] + cost
                    p[j] = i - 1
                j += 1
            if j >= n or load > w or cost > max_load:
                break
    return p, v[-1]


def route_retrieve(n, p, s):
    trip = [[] for i in range(n)]
    t = 0
    j = n
    while True:
        t += 1
        i = p[j]
        for k in range(i + 1, j + 1):
            trip[t].append(s[k])
        j = i
        if i == 0:
            break
    return trip


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
    return trip[:(t+1), :]


@njit
def get_max_route_len(q, w):
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
