import numpy as np
from numba import njit, int32
from usecase.dvrp.utils.route.angle import get_angle, near_neighbor
from usecase.dvrp.utils.heuristics.local_search.single_relocate import m1_cost_inter, do_m1_inter, do_m1_intra
from usecase.dvrp.utils.heuristics.local_search.double_relocate import m2_cost_inter, do_m2_inter, do_m2_intra
from usecase.dvrp.utils.heuristics.local_search.double_reverse_relocate import m3_cost_inter, do_m3_inter, do_m3_intra
from usecase.dvrp.utils.heuristics.local_search.single_inter_swap import m4_cost_inter, do_m4_inter
from usecase.dvrp.utils.heuristics.local_search.asm_inter_swap import m5_cost_inter, do_m5_inter
from usecase.dvrp.utils.route.repr import trip_lookup, trip_lookup_precedence


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


@njit(fastmath=True)
def descend(n, c, trip_dmd, q, w, lookup, neighbor, trip_num, lookup_prev, lookup_next):
    # lookup = trip_lookup(trip, n)
    gain = -1

    # todo: check the validity when moving elements into empty trip
    for i in range(len(trip_num)):
        if trip_num[i] == 0:
            for j in range(1, n + 1):
                r1 = lookup[j, 0]
                pos1 = lookup[j, 1]
                r2 = i
                pos2 = -1
                u = j
                u_prev = lookup_prev[u]
                x = lookup_next[u]
                x_post = lookup_next[x]
                v = 0
                y = 0
                u_dmd = q[u]
                x_dmd = q[x]
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > 0:
                    do_m1_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, trip_num, lookup_prev, lookup_next, u_prev, u, x,
                                v, y)
                    return gain
                if x:
                    gain2 = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
                    if gain2 > 0:
                        do_m2_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                    u_prev, u, x, x_post, v, y)
                        return gain2
                    gain3 = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
                    if gain3 > 0:
                        do_m3_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                    u_prev, u, x, x_post, v, y)
                        return gain3
            break

    for i, j in neighbor:
        r1 = lookup[i, 0]
        pos1 = lookup[i, 1]
        r2 = lookup[j, 0]
        pos2 = lookup[j, 1]
        u = i
        u_prev = lookup_prev[u]
        x = lookup_next[u]
        x_post = lookup_next[x]
        v = j
        v_prev = lookup_prev[v]
        y = lookup_next[v]
        u_dmd = q[u]
        x_dmd = q[x]
        v_dmd = q[v]
        y_dmd = q[y]

        if r1 != r2:  # inter route case
            if trip_dmd[r2] + u_dmd <= w:  # demand check for m1
                gain = m1_cost_inter(c, u_prev, u, x, v, y)  # 366 ns ± 8.05 ns
                if gain > 0:
                    do_m1_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, trip_num, lookup_prev, lookup_next, u_prev, u, x,
                                v, y)  # 773 ns ± 1.23 ns
                    return gain
            if x and trip_dmd[r2] + u_dmd + x_dmd <= w:  # u is not the last element
                gain2 = m2_cost_inter(c, u_prev, u, x, x_post, v, y)  # 462 ns ± 5.04 ns
                if gain2 > 0:
                    do_m2_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                u_prev, u, x, x_post, v, y)
                    return gain2
                gain3 = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain3 > 0:
                    do_m3_inter(r1, r2, pos2, lookup, trip_dmd, u_dmd, x_dmd, trip_num, lookup_prev, lookup_next,
                                u_prev, u, x, x_post, v, y)
                    return gain3

            if trip_dmd[r1] - u_dmd + v_dmd <= w and trip_dmd[r2] - v_dmd + u_dmd <= w:
                gain4 = m4_cost_inter(c, u_prev, u, x, v_prev, v, y)
                if gain4 > 0:
                    do_m4_inter(r1, r2, pos1, pos2, u_prev, u, x, v_prev, v, y, lookup, lookup_prev, lookup_next,
                                trip_dmd, u_dmd, v_dmd)
                    return gain4
            if x and trip_dmd[r1] - u_dmd - x_dmd + v_dmd <= w and trip_dmd[r2] - v_dmd + u_dmd + x_dmd <= w:
                gain5 = m5_cost_inter(c, u_prev, u, x, x_post, v_prev, v, y)
                if gain5 > 0:
                    do_m5_inter(r1, r2, pos1, pos2, lookup, trip_dmd, u_dmd, x_dmd, v_dmd, trip_num, lookup_prev,
                                lookup_next, u_prev, u, x, x_post, v_prev, v, y)
                    return gain5
        else:  # intra route case
            if u != y:
                gain = m1_cost_inter(c, u_prev, u, x, v, y)
                if gain > 0:
                    do_m1_intra(pos1, pos2, u_prev, u, x, v, y, lookup, lookup_next, lookup_prev)
                    return gain
            if x and x != v and u != y:  # u is not the last element (x not 0), (u x) and (v y) no overlap
                gain2 = m2_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain2 > 0:
                    do_m2_intra(pos1, pos2, u_prev, u, x, x_post, v, y, lookup, lookup_next, lookup_prev)
                    return gain2
                gain3 = m3_cost_inter(c, u_prev, u, x, x_post, v, y)
                if gain3 > 0:
                    do_m3_intra(pos1, pos2, u_prev, u, x, x_post, v, y, lookup, lookup_next, lookup_prev)
                    return gain3

            # pass
    return gain
# for testings
# if np.sum(trip_lookup(trip, n) - lookup) != 0:
#     raise ValueError("Move 1 error ")
