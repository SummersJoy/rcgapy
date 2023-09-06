import numpy as np
from numba import njit, int32


@njit
def trip_lookup(trip, n):
    """
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
