import numpy as np
import bisect
from numba import njit
from usecase.dvrp.utils.heuristics.local_search.single_relocate import m1_cost_inter, m1_cost_intra, do_m1_inter, \
    do_m1_intra
from usecase.dvrp.utils.heuristics.local_search.double_relocate import m2_cost_inter, do_m2_inter
from usecase.dvrp.utils.route.repr import trip_lookup, get_trip_len, trip_lookup_precedence, get_trip_num, \
    get_trip_dmd, lookup2trip
from utils.numba.random import bisect as nb_bisect
from usecase.dvrp.utils.heuristics.local_search.single_relocate import m1_lookup_inter_update, m1_lookup_intra_update


def test_nb_bisect():
    for _ in range(1000):
        res = np.sort(np.random.random(1000))
        val = np.random.random(1)
        control_val = bisect.bisect(res, val)
        test_val = nb_bisect(res, val)
        assert control_val == test_val + 1
    print(f"numba bisect test passed!")


def trip_test(trips, num):
    res = []
    for trip in trips:
        for j in trip:
            if j not in res:
                res.append(j)
    res = set(res)
    if len(res) != num + 1:
        full = set(range(1, num + 1))
        missing = full.difference(res)
        print(f"{missing} is missing")
        raise ValueError("Check trip construction function!")
    else:
        print("Trip test passed!")


@njit
def test_operation_m1(c, trip, n, q):
    """
    test 2 neighborhood search
    for empty route, client v is expressed as pos2=-1
    """
    trip_num = get_trip_num(trip)
    trip_dmd = get_trip_dmd(trip, q, trip_num)
    lookup = trip_lookup(trip, n)
    lookup_prev, lookup_next = trip_lookup_precedence(trip, trip_num, n)
    fitness = get_trip_len(c, trip)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
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
                if r1 != r2:
                    lookup_cpy = lookup.copy()
                    lookup_prev_cpy = lookup_prev.copy()
                    lookup_next_cpy = lookup_next.copy()
                    gain = m1_cost_inter(c, u_prev, u, x, v, y)
                    do_m1_inter(r1, r2, pos2, lookup_cpy, trip_dmd, u_dmd, trip_num, lookup_prev_cpy, lookup_next_cpy,
                                u_prev, u, x, v, y)
                    tmp = lookup2trip(lookup_cpy, n)
                    trip_num_test = get_trip_num(tmp)
                    lookup_prev_test, lookup_next_test = trip_lookup_precedence(tmp, trip_num_test, n)
                    # check if lookup is updated correctly
                    if abs(fitness - get_trip_len(c, tmp) - gain) < 1e-4:
                        print("lookup table -> trip and cost check passed!")
                    else:
                        print(f"m1 test inter route failed at nodes {i}, {j}, Cost difference: "
                              f"{fitness - get_trip_len(c, tmp) - gain}")
                        raise ValueError("m1 test failed")
                    if np.allclose(lookup_prev_test[1:], lookup_prev_cpy[1:]):
                        print("lookup_prev update test passed")
                    else:
                        raise ValueError("m1 test failed at lookup_prev test")
                    if np.allclose(lookup_next_test[1:], lookup_next_cpy[1:]):
                        print("lookup_next update test passed")
                    else:
                        raise ValueError("m1 test failed at lookup_prev test")
                else:
                    if pos2 + 1 != pos1:
                        lookup_cpy = lookup.copy()
                        lookup_prev_cpy = lookup_prev.copy()
                        lookup_next_cpy = lookup_next.copy()
                        gain = m1_cost_intra(c, u_prev, u, x, v, y)
                        do_m1_intra(pos1, pos2, u_prev, u, x, v, y, lookup_cpy, lookup_next_cpy, lookup_prev_cpy)
                        tmp = lookup2trip(lookup_cpy, n)
                        trip_num_test = get_trip_num(tmp)
                        lookup_prev_test, lookup_next_test = trip_lookup_precedence(tmp, trip_num_test, n)
                        if abs(fitness - get_trip_len(c, tmp) - gain) < 1e-4:
                            pass
                        else:
                            print(f"m1 test intra route failed at nodes {i}, {j},{pos1},{pos2}, Cost difference: "
                                  f"{fitness - get_trip_len(c, tmp) - gain}")
                        if np.allclose(lookup_prev_test[1:], lookup_prev_cpy[1:]):
                            print("lookup_prev update test passed")
                        else:
                            raise ValueError("m1 test failed at lookup_prev test")
                        if np.allclose(lookup_next_test[1:], lookup_next_cpy[1:]):
                            print("lookup_next update test passed")
                        else:
                            raise ValueError("m1 test failed at lookup_prev test")
    print("m1 operation test passed!")


def test_operation_m2(c, trip, n, q, trip_dmd):
    """
    test 2 neighborhood search
    for empty route, client v is expressed as pos2=-1
    """
    lookup = trip_lookup(trip, n)
    fitness = get_trip_len(c, trip)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                r1 = lookup[i, 0]
                pos1 = lookup[i, 1]
                r2 = lookup[j, 0]
                pos2 = lookup[j, 1]
                print(f"r1: {r1}, r2: {r2}, pos1: {pos1}, pos2: {pos2}")
                if r1 != r2 and trip[r1, pos1 + 1] != 0:
                    gain = m2_cost_inter(c, r1, r2, pos1, pos2, trip)
                    tmp = trip.copy()
                    do_m2_inter(r1, r2, pos1, pos2, tmp, lookup, trip_dmd, q)
                    if not np.allclose(trip_lookup(tmp, n), lookup):
                        print(f"lookup error with difference {trip_lookup(tmp, n) - lookup}")
                        raise ValueError("lookup")
                    lookup = trip_lookup(trip, n)
                    if abs(fitness - get_trip_len(c, tmp) - gain) < 1e-4:
                        pass
                    else:
                        print(f"m2 test inter route failed at nodes {i}, {j}, Cost difference: "
                              f"{fitness - get_trip_len(c, tmp) - gain}")
                else:
                    pass
                    # gain = m1_cost_intra(c, r1, pos1, pos2, trip)
                    # tmp = trip.copy()
                    # do_m1_intra(r1, pos1, pos2, tmp)
                    # if abs(fitness - get_trip_len(c, tmp) - gain) < 1e-4:
                    #     pass
                    # else:
                    #     print(f"m2 test intra route failed at nodes {i}, {j},{pos1},{pos2}, Cost difference: "
                    #           f"{fitness - get_trip_len(c, tmp) - gain}")
    print("m1 operation test passed!")


def test_lookup(trip, n):
    lookup = trip_lookup(trip, n)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                r1 = lookup[i, 0]
                pos1 = lookup[i, 1]
                r2 = lookup[j, 0]
                pos2 = lookup[j, 1]
                if r1 != r2:
                    m1_lookup_inter_update(r2, pos2, u, v, lookup, lookup_next)
                    do_m1_inter(r1, r2, pos1, pos2, trip)
                    lookup1 = trip_lookup(trip, n)
                    if not np.allclose(lookup1, lookup):
                        print("lookup test inter failed")
                        raise ValueError("failed")
                else:
                    m1_lookup_intra_update(trip, r1, pos1, pos2, lookup)
                    do_m1_intra(r1, pos1, pos2, trip)
                    lookup1 = trip_lookup(trip, n)
                    if not np.allclose(lookup1, lookup):
                        print("lookup test intra failed")
                        raise ValueError("failed")
