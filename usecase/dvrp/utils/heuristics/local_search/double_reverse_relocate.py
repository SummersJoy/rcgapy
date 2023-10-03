from numba import njit, int32
import numpy as np


@njit(fastmath=True)
def m3_cost_inter(c, u_prev, u, x, x_post, v, y):
    route1_break = c[u_prev, u] + c[u, x] + c[x, x_post]
    route1_repair = c[u_prev, x_post]
    route1_gain = route1_break - route1_repair
    route2_break = c[v, y]
    route2_repair = c[v, x] + c[x, u] + c[u, y]
    route2_gain = route2_break - route2_repair
    gain = route1_gain + route2_gain
    return gain


@njit
def do_m3_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next, u_prev, u, x, x_post,
                v, y) -> None:
    m3_lookup_inter_update(r2, pos2, u, x, v, lookup, lookup_next)
    # update lookup precedence for T(u)
    lookup_next[u_prev] = x_post
    lookup_prev[x_post] = u_prev

    # update lookup precedence for T(v)
    lookup_next[v] = x
    lookup_prev[x] = v
    lookup_next[u] = y
    lookup_prev[y] = u
    lookup_next[x] = u
    lookup_prev[u] = x

    # update route demand
    dmd = u_dmd + x_dmd
    trip_dmd[r1] -= dmd
    trip_dmd[r2] += dmd

    # update trip_num
    trip_num[r1] -= 2
    trip_num[r2] += 2


@njit
def m3_lookup_inter_update(r2: int, pos2: int, u: int, x: int, v: int, lookup: np.ndarray, lookup_next: np.ndarray):
    """
    update move 3 trip lookup table after inter route relocation
    """
    # update route1 T(u)
    cust = lookup_next[x]
    while cust:
        lookup[cust, 1] -= 2
        cust = lookup_next[cust]
    # update route 2 T(v)
    cust = lookup_next[v]
    while cust:
        lookup[cust, 1] += 2
        cust = lookup_next[cust]
    # update u
    lookup[x, 0] = r2
    lookup[x, 1] = pos2 + 1
    lookup[u, 0] = r2
    lookup[u, 1] = pos2 + 2
