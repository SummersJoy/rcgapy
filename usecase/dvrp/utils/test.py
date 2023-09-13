import numpy as np
import bisect
from numba import njit
from usecase.dvrp.utils.heuristics.local_search.single_relocate import m1_cost_inter, m1_cost_intra, do_m1_inter, \
    do_m1_intra
from usecase.dvrp.utils.heuristics.local_search.double_relocate import m2_cost_inter, do_m2_inter
from usecase.dvrp.utils.route.repr import trip_lookup, get_trip_len
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
def test_operation_m1(c, trip, n):
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
                if r1 != r2:
                    gain = m1_cost_inter(c, r1, r2, pos1, pos2, trip)
                    tmp = trip.copy()
                    do_m1_inter(r1, r2, pos1, pos2, tmp)
                    if abs(fitness - get_trip_len(c, tmp) - gain) < 1e-4:
                        pass
                    else:
                        print(f"m1 test inter route failed at nodes {i}, {j}, Cost difference: "
                              f"{fitness - get_trip_len(c, tmp) - gain}")
                else:
                    gain = m1_cost_intra(c, r1, pos1, pos2, trip)
                    tmp = trip.copy()
                    do_m1_intra(r1, pos1, pos2, tmp)
                    if abs(fitness - get_trip_len(c, tmp) - gain) < 1e-4:
                        pass
                    else:
                        print(f"m1 test intra route failed at nodes {i}, {j},{pos1},{pos2}, Cost difference: "
                              f"{fitness - get_trip_len(c, tmp) - gain}")
    print("m1 operation test passed!")


@njit
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
                if r1 != r2 and trip[r1, pos1 + 1]!= 0:
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
                    m1_lookup_inter_update(trip, r1, r2, pos1, pos2, lookup)
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
