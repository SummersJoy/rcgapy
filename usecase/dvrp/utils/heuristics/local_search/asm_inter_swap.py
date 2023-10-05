from numba import njit
import numpy as np


@njit(fastmath=True)
def m5_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y):
    route1_break = c[u_prev, u] + c[u, x] + c[x, x_post]
    route1_repair = c[u_prev, v] + c[v, x_post]
    route1_gain = route1_break - route1_repair
    route2_break = c[v_prev, v] + c[v, y]
    route2_repair = c[v_prev, u] + c[u, x] + c[x, y]
    route2_gain = route2_break - route2_repair
    gain = route1_gain + route2_gain
    return gain


@njit
def do_m5_inter(r1, r2, pos1, pos2, lookup, trip_dmd, u_dmd, x_dmd, v_dmd, trip_num, lookup_prev, lookup_next, u_prev,
                u, x, x_post, v_prev, v, y) -> None:
    m5_lookup_inter_update(r1, r2, pos1, pos2, u, x, v, lookup, lookup_next)
    # update lookup precedence for T(u)
    lookup_next[u_prev] = v
    lookup_prev[v] = u_prev
    lookup_next[v] = x_post
    lookup_prev[x_post] = v

    # update lookup precedence for T(v)
    lookup_next[v_prev] = u
    lookup_prev[u] = v_prev
    lookup_next[x] = y
    lookup_prev[y] = x

    # update route demand
    dmd = u_dmd + x_dmd - v_dmd
    trip_dmd[r1] -= dmd
    trip_dmd[r2] += dmd

    # update trip_num
    trip_num[r1] -= 1
    trip_num[r2] += 1


@njit
def m5_lookup_inter_update(r1: int, r2: int, pos1: int, pos2: int, u: int, x: int, v: int, lookup: np.ndarray,
                           lookup_next: np.ndarray) -> None:
    """
    update move 5 trip lookup table after inter route relocation
    """
    # update route1 T(u)
    cust = lookup_next[x]
    while cust:
        lookup[cust, 1] -= 1
        cust = lookup_next[cust]
    # update route 2 T(v)
    cust = lookup_next[v] if v else 0
    while cust:
        lookup[cust, 1] += 1
        cust = lookup_next[cust]
    # update u
    lookup[u, 0] = r2
    lookup[u, 1] = pos2
    lookup[x, 0] = r2
    lookup[x, 1] = pos2 + 1
    lookup[v, 0] = r1
    lookup[v, 1] = pos1


@njit
def do_m5_intra(pos1, pos2, u_prev, u, x, x_post, v, y, lookup, lookup_next, lookup_prev):
    # update lookup table
    m5_lookup_intra_update(pos1, pos2, u, x, v, y, lookup, lookup_next)
    # update lookup_prev, lookup_next
    m5_lookup_precedence_update(lookup_prev, lookup_next, u_prev, u, x, x_post, v, y)


@njit
def m5_lookup_intra_update(pos1: int, pos2: int, u: int, x: int, v: int, y: int, lookup: np.ndarray,
                           lookup_next: np.ndarray):
    """
    update trip lookup table after intra route relocation
    """
    if pos1 + 1 < pos2:
        cust = lookup_next[x]
        while cust != y:
            lookup[cust, 1] -= 2
            cust = lookup_next[cust]
        lookup[u, 1] = pos2 - 1
        lookup[x, 1] = pos2
    elif pos1 > pos2 + 1:
        cust = y
        while cust != u:
            lookup[cust, 1] += 2
            cust = lookup_next[cust]
        lookup[u, 1] = pos2 + 1
        lookup[x, 1] = pos2 + 2
    else:
        raise ValueError(f"Duplicated i, j: {u}")


@njit
def m5_lookup_precedence_update(lookup_prev, lookup_next, u_prev, u, x, x_post, v, y):
    """
    update lookup_prev and lookup_next after performing m1
    """

    # remove u
    lookup_next[u_prev] = x_post
    lookup_prev[x_post] = u_prev
    # insert u after v
    lookup_next[v] = u
    lookup_prev[u] = v
    lookup_next[x] = y
    lookup_prev[y] = x
