import numpy as np
from numba import njit, int32
from usecase.dvrp.utils.route.angle import get_angle, near_neighbor
from usecase.dvrp.utils.heuristics.local_search.single_relocate import m1_cost_inter, m1_cost_intra, do_m1_inter, \
    do_m1_intra
from usecase.dvrp.utils.heuristics.local_search.double_relocate import m2_cost_inter, do_m2_inter
from usecase.dvrp.utils.heuristics.local_search.single_inter_swap import m4_cost_inter, do_m4_inter
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
                u_prev = trip[r1, pos1 - 1] if pos1 >= 1 else 0
                u = j
                x = trip[r1, pos1 + 1]
                v = 0
                y = 0
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > 0:
                    do_m1_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, q[j])
                    return gain
            break

    for i, j in neighbor:
        r1 = lookup[i, 0]
        pos1 = lookup[i, 1]
        r2 = lookup[j, 0]
        pos2 = lookup[j, 1]
        u_prev = trip[r1, pos1 - 1] if pos1 >= 1 else 0
        u = i
        x = trip[r1, pos1 + 1]
        x_post = trip[r1, pos1 + 2]
        v_prev = trip[r2, pos2 - 1] if pos2 >= 1 else 0
        v = j
        y = trip[r2, pos2 + 1]
        u_dmd = q[u]
        x_dmd = q[x]
        v_dmd = q[v]
        y_dmd = q[y]

        if r1 != r2:  # inter route case
            if trip_dmd[r2] + u_dmd <= w:  # demand check for m1
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > 0:
                    do_m1_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, u_dmd)
                    return gain
            if trip[r1, pos1 + 1] and trip_dmd[r2] + u_dmd + x_dmd <= w:  # u is not the last element
                gain2 = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain2 > 0:
                    do_m2_inter(r1, r2, pos1, pos2, trip, lookup, trip_dmd, q)
                    return gain2
            if trip_dmd[r1] - u_dmd + v_dmd <= w and trip_dmd[r2] - v_dmd + u_dmd <= w:
                gain4 = m4_cost_inter(c, u_prev, u, x, v_prev, v, y)
                if gain4 > 0:
                    do_m4_inter(r1, r2, pos1, pos2, u, v, trip, lookup, trip_dmd, u_dmd, v_dmd)
                    return gain4
        else:  # intra route case
            gain = m1_cost_intra(c, r1, pos1, pos2, trip)
            if gain > 0:
                do_m1_intra(r1, pos1, pos2, trip, lookup)
                return gain
    return gain
# for testings
# if np.sum(trip_lookup(trip, n) - lookup) != 0:
#     raise ValueError("Move 1 error ")
