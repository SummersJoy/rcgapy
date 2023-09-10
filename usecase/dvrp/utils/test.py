import numpy as np
import bisect
from numba import njit
from usecase.dvrp.utils.heuristics.local_search.single_relocate import m1_cost_inter, m1_cost_intra, do_m1_inter, \
    do_m1_intra
from usecase.dvrp.utils.route.repr import trip_lookup
from utils.numba.random import bisect as nb_bisect


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
def get_route_len(c, route):
    res = c[0, route[0]]
    for i in range(len(route) - 1):
        if route[i] == 0:
            break
        else:
            res += c[route[i], route[i + 1]]
    return res


@njit
def get_trip_len(c, trip):
    res = 0
    for route in trip:
        res += get_route_len(c, route)
    return res


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