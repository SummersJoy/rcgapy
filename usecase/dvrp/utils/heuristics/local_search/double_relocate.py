from numba import njit, int32
import numpy as np


@njit
def m2_cost_inter(c, r1, r2, pos1, pos2, trip):
    u_prev = trip[r1, pos1 - 1] if pos1 >= 1 else 0
    u = trip[r1, pos1]
    x = trip[r1, pos1 + 1]
    x_post = trip[r1, pos1 + 2]
    route1_break = c[u_prev, u] + c[u, x] + c[x, x_post]
    route1_repair = c[u_prev, x_post]
    route1_gain = route1_break - route1_repair
    v = trip[r2, pos2]
    v_post = trip[r2, pos2 + 1]
    route2_break = c[v, v_post]
    route2_repair = c[v, u] + c[u, x] + c[x, v_post]
    route2_gain = route2_break - route2_repair
    gain = route1_gain + route2_gain
    return gain


@njit
def do_m2_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, q):
    m2_lookup_inter_update(trip, r1, r2, pos1, pos2, lookup)
    route_id1 = trip[r1]
    route_id2 = trip[r2]
    u = route_id1[pos1]
    x = route_id1[pos1 + 1]
    trip[r1] = np.concatenate((route_id1[:pos1], route_id1[pos1 + 2:], np.zeros(2)))
    trip[r2] = np.concatenate((route_id2[:pos2 + 1], u * np.ones(1), x * np.ones(1), route_id2[(pos2 + 1):-2]))
    # update route demand
    dmd = q[u] + q[x]
    trip_dmd[r1] -= dmd
    trip_dmd[r2] += dmd


@njit
def m2_lookup_inter_update(trip: np.ndarray, r1: int, r2: int, pos1: int, pos2: int, lookup: np.ndarray):
    """
    update move 2 trip lookup table after inter route relocation
    """
    u = trip[r1, pos1]
    x = trip[r1, pos1 + 1]
    n_rol, n_col = trip.shape
    for i in range(pos1 + 2, n_col):
        cust = trip[r1, i]
        if cust == 0:
            break
        lookup[cust, 1] -= 2

    for i in range(pos2 + 1, n_col):
        cust = trip[r2, i]
        if cust == 0:
            break
        lookup[cust, 1] += 2
    lookup[u, 0] = r2
    lookup[u, 1] = pos2 + 1
    lookup[x, 0] = r2
    lookup[x, 1] = pos2 + 2
