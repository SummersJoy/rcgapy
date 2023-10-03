import numpy as np
from numba import njit
from dvrp.utils.route.repr import trip_lookup


@njit(fastmath=True)
def m1_cost_inter(c, u_prev, u, x, v, y):
    route1_break1 = c[u_prev, u]
    route1_break2 = c[u, x]
    route1_repair = c[u_prev, x]
    route1_gain = route1_break1 + route1_break2 - route1_repair

    route2_break = c[v, y]
    route2_repair1 = c[v, u]
    route2_repair2 = c[u, y]
    route2_gain = route2_break - route2_repair1 - route2_repair2
    gain = route1_gain + route2_gain
    return gain


@njit(fastmath=True)
def m1_cost_intra(c, u_prev, u, x, v, y):
    if u == y:
        return 0
    return m1_cost_inter(c, u_prev, u, x, v, y)


@njit
def do_m1_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, trip_num, lookup_prev, lookup_next, u_prev, u, x, v, y):
    """
    perform m1 movements
    1. update lookup table
    2. update lookup_prev and lookup_next
    3. update trip total demand
    4. update number of customers in changed routes
    """
    # update lookup table
    m1_lookup_inter_update(r2, pos2, u, v, lookup, lookup_next)

    # update lookup_prev and lookup_next
    m1_lookup_precedence_update(lookup_prev, lookup_next, u_prev, u, x, v, y)
    # update route demand
    trip_dmd[r1] -= u_dmd
    trip_dmd[r2] += u_dmd
    # update number of customers in each route
    trip_num[r1] -= 1
    trip_num[r2] += 1


@njit
def do_m1_intra(pos1, pos2, u_prev, u, x, v, y, lookup, lookup_next, lookup_prev):
    # update lookup table
    m1_lookup_intra_update(pos1, pos2, u, v, y, lookup, lookup_next)
    # update lookup_prev, lookup_next
    m1_lookup_precedence_update(lookup_prev, lookup_next, u_prev, u, x, v, y)


@njit
def m1_lookup_inter_update(r2: int, pos2: int, u: int, v: int, lookup: np.ndarray, lookup_next: np.ndarray) -> None:
    """
    update trip lookup table after inter route relocation
    """
    # update route1 T(u)
    cust = lookup_next[u]
    while cust:
        lookup[cust, 1] -= 1
        cust = lookup_next[cust]
    # update route 2 T(v)
    cust = lookup_next[v] if v else 0  # if v is 0 -> empty route, then do not update T(v)
    while cust:
        lookup[cust, 1] += 1
        cust = lookup_next[cust]
    # update u
    lookup[u, 0] = r2
    lookup[u, 1] = pos2 + 1


@njit
def m1_lookup_intra_update(pos1: int, pos2: int, u: int, v: int, y: int, lookup: np.ndarray, lookup_next: np.ndarray):
    """
    update trip lookup table after intra route relocation
    """
    if pos1 < pos2:
        cust = lookup_next[u]
        while cust != y:
            lookup[cust, 1] -= 1
            cust = lookup_next[cust]
        lookup[u, 1] = pos2
    elif pos1 > pos2:
        cust = y
        while cust != u:
            lookup[cust, 1] += 1
            cust = lookup_next[cust]
        lookup[u, 1] = pos2 + 1
    else:
        raise ValueError(f"Duplicated i, j: {u}")


@njit
def do_m1(i, j, lookup, q, trip_dmd, w, c, trip):
    r1 = lookup[i, 0]
    pos1 = lookup[i, 1]
    r2 = lookup[j, 0]
    pos2 = lookup[j, 1]
    u_dmd = q[i]
    # demand check
    if trip_dmd[r2] + u_dmd > w:
        return -1
    if r1 != r2:  # inter route case
        gain = m1_cost_inter(c, r1, r2, pos1, pos2, trip)
        if gain > 0:
            do_m1_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, u_dmd)
            return gain
    else:  # intra route case
        gain = m1_cost_intra(c, r1, pos1, pos2, trip)
        if gain > 0:
            do_m1_intra(r1, pos1, pos2, trip, lookup)
            return gain


@njit
def m1_lookup_precedence_update(lookup_prev, lookup_next, u_prev, u, x, v, y):
    """
    update lookup_prev and lookup_next after performing m1
    """
    # remove u
    lookup_next[u_prev] = x
    lookup_prev[x] = u_prev
    # insert u after v
    lookup_next[v] = u
    lookup_prev[u] = v
    lookup_next[u] = y
    lookup_prev[y] = u
