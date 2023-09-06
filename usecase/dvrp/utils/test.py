from usecase.dvrp.utils.heuristics.local_search.operator import m1_cost_inter, do_m1_inter
from usecase.dvrp.utils.io.repr import trip_lookup


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


def get_route_len(c, route):
    res = c[0, route[0]]
    for i in range(len(route) - 1):
        if route[i] == 0:
            break
        else:
            res += c[route[i], route[i + 1]]
    return res


def get_trip_len(c, trip):
    res = 0
    for route in trip:
        res += get_route_len(c, route)
    return res


def test_operation_m1(c, trip, n):
    """
    test 2 neighborhood search
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
                        print(f"m1 test failed at nodes {i}, {j},Cost difference: {fitness - get_trip_len(c, tmp) - gain}")
