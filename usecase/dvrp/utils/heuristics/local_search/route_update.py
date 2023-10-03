from numba import njit
import numpy as np


@njit
def right_shift(trip: np.ndarray, val: int, rid: int, m: int, pos: int) -> None:
    """
    route extension
    m, the max_route_len + val
    """
    # for i in range(m, pos, -1):
    #     trip[rid, i] = trip[rid, i - val]
    total = pos + m
    for i in range(pos, m):
        j = total - i
        trip[rid, j] = trip[rid, j - val]


@njit
def fill_elements():
    pass


@njit
def left_shift(trip: np.ndarray, val: int, rid: int, m: int, pos: int) -> None:
    """
    remove elements from route
    """
    for i in range(pos, m):
        trip[rid, i] = trip[rid, i + val]
