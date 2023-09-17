from numba import njit, int32
import numpy as np
from usecase.dvrp.utils.route.angle import get_angle


class Sweep:
    def __init__(self, cx, cy, q, c, max_route_len):
        self.cx = cx
        self.cy = cy
        self.q = q
        self.c = c
        self.max_route_len = max_route_len

    @njit
    def __sweep_constractor(self):
        return sweep_constructor(self.cx, self.cy, self.q, self.c, self.max_route_len)


@njit
def sweep_constructor(cx, cy, q, c, max_route_len, w, max_dist):
    n = len(cx) - 1
    agl = get_angle(cx, cy)
    seq = np.argsort(agl[1:]) + 1
    seed = np.random.randint(0, 50)
    seq = np.concatenate((seq[seed:], seq[:seed]))
    trip = np.zeros((n, max_route_len), dtype=int32)
    cap = np.zeros(n)
    dist = np.zeros(n)
    dist_prev = np.zeros(n)
    route_idx = 0
    pos = 0
    prev = 0
    for i in range(n):
        cust = seq[i]
        cap[route_idx] += q[i]
        dist_prev[route_idx] += c[prev, cust]
        dist[route_idx] = dist_prev[route_idx] + c[cust, 0]
        if cap[route_idx] <= w and dist[route_idx] <= max_dist:
            trip[route_idx, pos] = cust
            pos += 1
        else:
            route_idx += 1
            pos = 0
            trip[route_idx, pos] = cust
            pos += 1
    return trip[:route_idx + 1]


@njit
def sweep_modifier(trip, k, c, angle, avr, mtl):
    avr = get_avr(c)


@njit(fastmath=True)
def get_avr(c):
    return np.mean(c[0:])


@njit(fastmath=True)
def get_delete_loc(trip, k, c, angle, avr, mtl):
    best = np.inf
    idx = 0
    for i in range(mtl):
        cust = trip[k, i]
        if cust == 0:
            break
        val = c[cust, 0] + angle[cust] * avr
        if val < best:
            best = val
            idx = i
    return idx
