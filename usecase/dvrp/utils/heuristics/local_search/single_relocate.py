import numpy as np
from numba import njit


@njit
def m1_cost_inter(c, r1, r2, pos1, pos2, trip):
    route_id1 = trip[r1]
    route_id2 = trip[r2]
    u_prev = route_id1[pos1 - 1] if pos1 >= 1 else 0
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


@njit
def m1_cost_intra(c, r, pos1, pos2, trip):
    if pos2 + 1 == pos1:
        return 0
    route = trip[r]
    u_prev = route[pos1 - 1] if pos1 >= 1 else 0
    u = route[pos1]
    u_post = route[pos1 + 1]
    u_break1 = c[u_prev, u]
    u_break2 = c[u, u_post]
    u_repair = c[u_prev, u_post]
    u_gain = u_break1 + u_break2 - u_repair
    v = route[pos2]
    v_post = route[pos2 + 1]
    v_break = c[v, v_post]
    v_repair1 = c[v, u]
    v_repair2 = c[u, v_post]
    v_gain = v_break - v_repair1 - v_repair2
    gain = u_gain + v_gain
    return gain


@njit
def do_m1_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, u_dmd):
    # update lookup table
    m1_lookup_inter_update(trip, r1, r2, pos1, pos2, lookup)
    # update trip routes
    route_id1 = trip[r1]
    route_id2 = trip[r2]
    u = route_id1[pos1]
    trip[r1] = np.concatenate((route_id1[:pos1], route_id1[pos1 + 1:], np.zeros(1)))
    trip[r2] = np.concatenate((route_id2[:pos2 + 1], u * np.ones(1), route_id2[(pos2 + 1):-1]))
    # update route demand
    trip_dmd[r1] -= u_dmd
    trip_dmd[r2] += u_dmd


@njit
def do_m1_intra(r, pos1, pos2, trip, lookup):
    m1_lookup_intra_update(trip, r, pos1, pos2, lookup)
    route = trip[r]
    if pos1 < pos2:
        trip[r] = np.concatenate(
            (route[:pos1], route[(pos1 + 1):(pos2 + 1)], route[pos1] * np.ones(1), route[pos2 + 1:]))
    else:
        trip[r] = np.concatenate(
            (route[:pos2 + 1], route[pos1] * np.ones(1), route[(pos2 + 1):pos1], route[pos1 + 1:]))


@njit
def m1_lookup_inter_update(trip: np.ndarray, r1: int, r2: int, pos1: int, pos2: int, lookup: np.ndarray):
    """
    update trip lookup table after inter route relocation
    """
    u = trip[r1, pos1]
    n_rol, n_col = trip.shape
    for i in range(pos1 + 1, n_col):
        cust = trip[r1, i]
        if cust == 0:
            break
        lookup[cust, 1] -= 1
    # for cust in trip[r1, (pos1 + 1):]:
    #     if cust == 0:
    #         break
    #     lookup[cust, 1] -= 1
    # for cust in trip[r2, (pos2 + 1):]:
    for i in range(pos2 + 1, n_col):
        cust = trip[r2, i]
        if cust == 0:
            break
        lookup[cust, 1] += 1
    lookup[u, 0] = r2
    lookup[u, 1] = pos2 + 1


@njit
def m1_lookup_intra_update(trip: np.ndarray, r1: int, pos1: int, pos2: int, lookup: np.ndarray):
    """
    update trip lookup table after intra route relocation
    """
    u = trip[r1, pos1]
    if pos1 < pos2:
        # for cust in trip[r1, (pos1 + 1):(pos2+1)]:
        for i in range(pos1 + 1, pos2 + 1):
            cust = trip[r1, i]
            if cust == 0:
                break
            lookup[cust, 1] -= 1
        lookup[u, 1] = pos2
    elif pos1 > pos2:
        lookup[u, 1] = pos2 + 1
        # for cust in trip[r1, (pos2 + 1):pos1]:
        for i in range(pos2 + 1, pos1):
            cust = trip[r1, i]
            lookup[cust, 1] += 1
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

