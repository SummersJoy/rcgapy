import numpy as np
from numba import njit, int32


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
