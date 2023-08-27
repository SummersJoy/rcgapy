# import numpy as np
# from numba import njit
#
#
# @njit(fastmath=True)
# def objective_function(x):
#     r = (x[0] - 1.) ** 2 + (x[1] - 2.) ** 2 + (x[2] - 3.) ** 2 + (x[3] - 1.) ** 2 + (x[4] - 1.) ** 2 + \
#         (x[5] - 1.) ** 2 - np.log(x[6] + 1.)
#     return r
#
#
# @njit(fastmath=True)
# def nonlinear_functions(x):
#     const1 = x[5] ** 2 + x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 5.5
#     const2 = x[1] ** 2 + x[4] ** 2 - 1.64
#     const3 = x[2] ** 2 + x[5] ** 2 - 4.25
#     const4 = x[2] ** 2 + x[4] ** 2 - 4.64
#     return const1, const2, const3, const4
#
