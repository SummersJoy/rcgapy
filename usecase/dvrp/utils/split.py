import numpy as np
from numba import njit

n = 10
s = np.random.permutation(10) + 1
q = np.random.randint(1, 10, 11)
q[0] = 0
d = np.zeros(11)
d[0] = 0
c = np.random.random((11, 11))
w = 20
max_load = 100


@njit
def split(n, s, q, d, c, w, max_load):
    v = np.empty(n + 1)
    p = np.empty(n, dtype=int)
    v[0] = 0
    v[1:] = np.inf
    for i in range(n):
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
    return p


def route_retrieve(n, p, s):
    trip = [[] for i in range(n)]
    t = 0
    j = n - 1
    while True:
        t += 1
        i = p[j]
        for k in range(i + 1, j + 1):
            trip[t].append(s[k])
        j = i
        if i == 0:
            break
