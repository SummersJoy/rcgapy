import numpy as np

from numba import njit, float64
from timeit import timeit

rain = np.random.random(size=10000000)


# pure Python implementation
def abc_model_py(a, b, c, rain):
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_in
        state_in = state_out
    return outflow


# numba version of the ABC-model
@njit(['float64[:](float64,float64,float64,float64[:])'])
def abc_model_numba(a, b, c, rain):
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_in
        state_in = state_out
    return outflow
