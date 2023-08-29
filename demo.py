import numpy as np
import time
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.model.optimize import _opt
from utils.visualize.dynamic import ga_dynamic_single


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
best_obj, best_ind, avg_fit, best_fit, pmt = _opt(objective_function, x_cts, x_int, lb_cts, ub_cts, lb_int,
                                                  ub_int, lin_lhs, lin_rhs, nonlinear_functions)
print(f"GA takes {time.perf_counter() - start} seconds to execute. ")

x = np.linspace(-5, 5, 101)
y = np.linspace(-5, 5, 101)
# full coordinate arrays
xx, yy = np.meshgrid(x, y)
var = np.empty((101, 2))
var[:, 0] = x
var[:, 1] = y
zz = np.empty((101, 101))
for i in range(101):
    for j in range(101):
        val = np.array([x[i], y[j]])
        zz[i, j] = objective_function(val)

fig = plt.figure()
h = plt.contourf(x, y, zz)


def animate(iter_id):
    global h
    plt.clf()
    h = plt.contourf(x, y, zz)
    for i in range(pmt.shape[0]):
        plt.plot(pmt[i, 0, iter_id], pmt[i, 1, iter_id], "*", color="red")
    plt.title(f"Generation Number: {iter_id}")
    return h


anim = FuncAnimation(fig, animate, frames=pmt.shape[-1], interval=100, repeat=False)
#
anim.save("./population.gif", writer="imagemagick", fps=5)
# animation = ga_dynamic_single(avg_fit, best_fit)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, zz)
# plt.show()
