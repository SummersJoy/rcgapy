import numpy as np
from numba import njit, int32
from usecase.dvrp.utils.route.angle import get_angle, near_neighbor
from usecase.dvrp.utils.heuristics.local_search.single_relocate import m1_cost_inter, m1_cost_intra, do_m1_inter, \
    do_m1_intra
from usecase.dvrp.utils.heuristics.local_search.double_relocate import m2_cost_inter, do_m2_inter
from usecase.dvrp.utils.route.repr import trip_lookup


@njit
def neighbourhood_gen(cx, cy, max_agl):
    angle = get_angle(cx, cy)
    nn = near_neighbor(angle, max_agl)
    n, m = nn.shape
    res = np.empty((n * m, 2), dtype=int32)
    count = 0
    for i in range(1, n):
        for j in range(m):
            val = nn[i, j]
            if val == 0:
                break
            else:
                res[count, 0] = i
                res[count, 1] = val
                count += 1
    return res[:count]


@njit
def descend(trip, n, c, trip_dmd, q, w, lookup, neighbor):
    # lookup = trip_lookup(trip, n)
    gain = -1

    for i in range(len(trip)):
        if trip[i, 0] == 0:
            for j in range(1, n + 1):
                r1 = lookup[j, 0]
                pos1 = lookup[j, 1]
                r2 = i
                pos2 = -1
                gain = m1_cost_inter(c, r1, r2, pos1, pos2, trip)
                if gain > 0:
                    do_m1_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, q[j])
                    return gain
            break

    for i, j in neighbor:
        r1 = lookup[i, 0]
        pos1 = lookup[i, 1]
        r2 = lookup[j, 0]
        pos2 = lookup[j, 1]
        u_dmd = q[i]
        if trip_dmd[r2] + u_dmd > w:
            continue
        if r1 != r2:  # inter route case
            gain = m1_cost_inter(c, r1, r2, pos1, pos2, trip)
            if gain > 0:
                do_m1_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, u_dmd)
                return gain
            gain2 = m2_cost_inter(c, r1, r2, pos1, pos2, trip)
            if gain2 > 0:
                do_m2_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, q)
                # if np.sum(trip_lookup(trip, n) - lookup) != 0:
                #     raise ValueError("Move 2 error ")
                return gain2
        else:  # intra route case
            gain = m1_cost_intra(c, r1, pos1, pos2, trip)
            if gain > 0:
                do_m1_intra(r1, pos1, pos2, trip, lookup)
                return gain
    return gain
