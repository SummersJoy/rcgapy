import numpy as np
import time
from utils.model.optimize import opt
from utils.visualize.dynamic import ga_dynamic_single

# setup
x_cts = np.array([0, 1, 2])
x_int = np.array([3, 4, 5, 6])
lb_cts = np.array([0., 0., 0.])
ub_cts = np.array([1.2, 1.8, 2.5])
lb_int = np.array([0, 0, 0, 0])
ub_int = np.array([1, 1, 1, 1])
size = 700
lin_rhs = np.array([[5., 1.2, 1.8, 2.5, 1.2]]).T
lin_lhs = np.array([
    [1., 1., 1., 1., 1., 1., 0.],
    [1., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 0., 1., 0., 0.],
    [0., 0., 1., 0., 0., 1., 0.],
    [1., 0., 0., 0., 0., 0., 1.]
])
non_rhs = np.array([5.5, 1.64, 4.25, 4.64])

# parameters
x_prob = 0.8  # cross over probability
m_prob = 0.005  # mutation probability
a = 0  # laplace crossover a
b_cts = 0.15  # laplace crossover real variable b
b_int = 0.35  # laplace crossover int variable b
p_cts = 10  # power mutation real variable
p_int = 4  # power mutation integer variable

max_iter = 1000
max_stall = 20
max_run = 16

# optimize
start = time.perf_counter()
best_obj, best_ind, avg_fit, best_fit = opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, non_rhs,
                                            x_prob, a, b_cts,
                                            b_int, m_prob, p_cts, p_int, size, max_iter, max_stall, max_run)
global_best_obj = 3.557463
print(f"Gap: {np.round((best_obj - global_best_obj) / global_best_obj, 6)}")
print(f"GA takes {time.perf_counter() - start} seconds to execute. ")
# # check for constraint violation

# violation = constraint_violation(np.array([0.2, 1.8, 1.90787, 1, 0, 0, 1]).reshape(7, 1), lin_lhs, lin_rhs, non_rhs)
# objective = objective_function(np.array([0.2, 1.28, 1.954483, 1, 0, 0, 1]).reshape(7, 1))
# violation = constraint_violation(best_ind.reshape(7, 1), lin_lhs, lin_rhs, non_rhs)

animation = ga_dynamic_single(avg_fit, best_fit)
