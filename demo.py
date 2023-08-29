import numpy as np
import time
from numba import njit
from utils.model.optimize import opt_single
from utils.visualize.static import plot_3d
from utils.visualize.dynamic import evolve_process


# define objective function and nonlinear functions
@njit(fastmath=True)
def objective_function(x):
    r = 3. * (1. - x[0]) ** 2 * np.exp(-(x[0] ** 2) - (x[1] + 1.) ** 2) - 10. * (
            x[0] / 5. - x[0] ** 3 - x[1] ** 5) * np.exp(-x[0] ** 2 - x[1] ** 2) - 1. / 3. * np.exp(
        -(x[0] + 1.) ** 2 - x[1] ** 2)
    return r


@njit(fastmath=True)
def nonlinear_functions(x):
    return (0.,)


# problem setup
x_cts = np.array([0, 1])  # index of continuous variables
x_int = np.array([2])  # index of integer variables
lb_cts = np.array([-3., -3.])  # lower bound of continuous variables
ub_cts = np.array([3., 3.])  # upper bound of continuous variables
lb_int = np.array([0])  # lower bound of integer variables
ub_int = np.array([1])  # upper bound of integer variables
lin_rhs = np.array([[6.]]).T  # right-hand side of linear inequality constraints
lin_lhs = np.array([  # left-hand side of linear inequality constraints
    [1., 0., 0.],
])

# optimize
start = time.perf_counter()
best_obj, best_ind, avg_fit, best_fit, pmt = opt_single(objective_function, x_cts, x_int, lb_cts, ub_cts, lb_int,
                                                        ub_int, lin_lhs, lin_rhs, nonlinear_functions)
print(f"GA takes {time.perf_counter() - start} seconds to execute. ")

anim = evolve_process(objective_function, pmt)
#
anim.save("./population.gif", writer="imagemagick", fps=5)

# show surface of peak funtion
plot_3d(objective_function)
