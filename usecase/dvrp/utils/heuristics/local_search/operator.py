import numpy as np
from numba import njit


def m1_cost_inter(c, r1, r2, pos1, pos2, trip):
    route_id1 = trip[r1]
    route_id2 = trip[r2]
    u_prev = route_id1[pos1 - 1] if pos1 > 1 else 0
    u = route_id1[pos1]
    u_post = route_id1[pos1 + 1]
    route1_break1 = c[u_prev, u]
    route1_break2 = c[u, u_post]
    route1_repair = c[u_prev, u_post]
    route1_gain = route1_break1 + route1_break2 - route1_repair
    v = route_id2[pos2]
    v_post = route_id2[pos2 + 1]
    route2_break = c[v, v_post]
    route2_repair1 = c[v, u]
    route2_repair2 = c[u, v_post]
    route2_gain = route2_break - route2_repair1 - route2_repair2
    gain = route1_gain + route2_gain
    return gain


def do_m1_inter(r1, r2, pos1, pos2, trip):
    route_id1 = trip[r1]
    route_id2 = trip[r2]
    u = route_id1[pos1]
    trip[r1] = np.concatenate((route_id1[:pos1], route_id1[pos1 + 1:], np.zeros(1)))
    trip[r2] = np.concatenate((route_id2[:pos2 + 1], u * np.ones(1), route_id2[(pos2 + 1):-1]))
