import numpy as np
import time
from utils.model.optimize import opt
from utils.visualize.dynamic import ga_dynamic_single

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
best_obj, best_ind, avg_fit, best_fit, violation = opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs)
global_best_obj = 3.557463
print(f"Gap: {np.round((best_obj - global_best_obj) / global_best_obj, 6)}, total violation: {violation}")
print(f"GA takes {time.perf_counter() - start} seconds to execute. ")

animation = ga_dynamic_single(avg_fit, best_fit)
animation.save("./problem8.gif", writer="ffmpeg", fps=60)
# num_trial = 100
# bst = np.empty(num_trial)
# for i in range(num_trial):
#     best_obj, best_ind, avg_fit, best_fit = opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, x_prob,
#                                                 a, b_cts, b_int, m_prob, p_cts, p_int, size, max_iter, max_stall,
#                                                 max_run)
#     bst[i] = best_obj
# print(f"The worst solution in {num_trial} run is {max(bst)}.")
