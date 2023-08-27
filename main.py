import numpy as np
import time
from numba import njit
from utils.model.optimize import opt
from utils.visualize.dynamic import ga_dynamic_single


# define objective function and nonlinear functions
@njit(fastmath=True)
def objective_function(x):
    r = (x[0] - 1.) ** 2 + (x[1] - 2.) ** 2 + (x[2] - 3.) ** 2 + (x[3] - 1.) ** 2 + (x[4] - 1.) ** 2 + \
        (x[5] - 1.) ** 2 - np.log(x[6] + 1.)
    return r


@njit(fastmath=True)
def nonlinear_functions(x):
    const1 = x[5] ** 2 + x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 5.5
    const2 = x[1] ** 2 + x[4] ** 2 - 1.64
    const3 = x[2] ** 2 + x[5] ** 2 - 4.25
    const4 = x[2] ** 2 + x[4] ** 2 - 4.64
    return const1, const2, const3, const4


# problem setup
x_cts = np.array([0, 1, 2])  # index of continuous variables
x_int = np.array([3, 4, 5, 6])  # index of integer variables
lb_cts = np.array([0., 0., 0.])  # lower bound of continuous variables
ub_cts = np.array([1.2, 1.8, 2.5])  # upper bound of continuous variables
lb_int = np.array([0, 0, 0, 0])  # lower bound of integer variables
ub_int = np.array([1, 1, 1, 1])  # upper bound of integer variables
lin_rhs = np.array([[5., 1.2, 1.8, 2.5, 1.2]]).T  # right-hand side of linear inequality constraints
lin_lhs = np.array([  # left-hand side of linear inequality constraints
    [1., 1., 1., 1., 1., 1., 0.],
    [1., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 1., 0., 0.],
    [0., 0., 1., 0., 0., 1., 0.],
    [1., 0., 0., 0., 0., 0., 1.]
])

# optimize
start = time.perf_counter()
best_obj, best_ind, avg_fit, best_fit, violation = opt(objective_function, x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int,
                                                       lin_lhs, lin_rhs, nonlinear_functions)
global_best_obj = 3.557463
print(f"Gap: {np.round((best_obj - global_best_obj) / global_best_obj, 6)}, total violation: {violation}")
print(f"GA takes {time.perf_counter() - start} seconds to execute. ")

animation = ga_dynamic_single(avg_fit, best_fit)
# animation.save("./problem8.gif", writer="ffmpeg", fps=60)
# num_trial = 100
# bst = np.empty(num_trial)
# for i in range(num_trial):
#     best_obj, best_ind, avg_fit, best_fit = opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, x_prob,
#                                                 a, b_cts, b_int, m_prob, p_cts, p_int, size, max_iter, max_stall,
#                                                 max_run)
#     bst[i] = best_obj
# print(f"The worst solution in {num_trial} run is {max(bst)}.")
