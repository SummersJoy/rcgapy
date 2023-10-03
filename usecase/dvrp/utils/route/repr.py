import numpy as np
from numba import njit, int32


@njit
def get_route_len(c, route):
    res = c[0, route[0]]
    i = 0
    for i in range(len(route) - 1):
        if route[i] == 0:
            break
        else:
            res += c[route[i], route[i + 1]]
    res += c[route[i + 1], 0]
    return res


@njit
def get_trip_len(c, trip):
    res = 0
    for route in trip:
        res += get_route_len(c, route)
    return res


@njit
def trip_lookup(trip: np.ndarray, n: int) -> np.ndarray:
    """
    Matrix representation of trips
    eg: res[2, 0] = 4, res[2, 1] = 1
    res[:, 0] -> trip_id, res[:, 1] -> position_id
    customer 2 is in trip 4, location 1
    n: number of customers
    """
    res = np.empty((n + 1, 2), dtype=int32)
    res[0:] = -1
    for route_id, route in enumerate(trip):
        for cust_id, cust in enumerate(route):
            if cust == 0:
                break
            res[cust, 0] = route_id
            res[cust, 1] = cust_id
    return res


@njit
def decoding(trip: np.ndarray, n) -> np.ndarray:
    """
    decode a trip into chromosome
    """
    res = np.empty(n + 1, dtype=int32)
    res[0] = 0
    count = 1
    for route in trip[::-1, :]:
        for cust in route:
            res[count] = cust
            if cust == 0:
                break
            count += 1
    return res


@njit(fastmath=True)
def get_route_dmd(route: np.ndarray, q: np.ndarray) -> float:
    """
    total demand
    """
    return sum(q[route])


@njit(fastmath=True)
def get_trip_dmd(trip: np.ndarray, q: np.ndarray, trip_num: np.ndarray) -> np.ndarray:
    n = len(trip)
    res = np.empty(n)
    for i in range(n):
        demand = 0.
        for j in range(trip_num[i]):
            demand += q[trip[i, j]]
        res[i] = demand
    return res


@njit
def get_trip_num(trip: np.ndarray) -> np.ndarray:
    """
    compute the number of customers on each trip
    """
    n = len(trip)
    res = np.empty(n, dtype=int32)
    for i in range(n):
        count = 0
        for j in trip[i]:
            if j == 0:
                break
            else:
                count += 1
        res[i] = count
    return res


@njit
def trip_lookup_precedence(trip: np.ndarray, trip_num: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    compute lookup_prev and lookup_next
    """
    lookup_next = np.zeros((n + 1), dtype=int32)
    lookup_prev = np.zeros((n + 1), dtype=int32)
    for i in range(len(trip)):
        for j in range(trip_num[i] - 1):
            target = trip[i, j]
            t_next = trip[i, j + 1]
            lookup_next[target] = t_next
            lookup_prev[t_next] = target
    return lookup_prev, lookup_next


@njit
def lookup2trip(lookup: np.ndarray, n: int, max_route_len: int, m: int) -> np.ndarray:
    """
    retrieve trip variable from lookup table
    """
    res = np.zeros((m, max_route_len), dtype=int32)
    max_rid = 0
    for i in range(1, len(lookup)):
        rid, pos = lookup[i]
        if rid > max_rid:
            max_rid = rid
        res[rid, pos] = i
    return res[:(max_rid + 1)]
