import matplotlib.pyplot as plt
import numpy as np


def ga_plot_single(avg, bst):
    npts = len(avg)
    x = list(range(npts))
    plt.plot(x, avg, ".")
    plt.plot(x, bst, "*")

    plt.xlabel("Generation")
    plt.ylabel("Fitness value")
    plt.title("Genetic Algorithm Numba")
    plt.show()


def plot_3d(objective_function):
    x = np.linspace(-5, 5, 101)
    y = np.linspace(-5, 5, 101)
    xx, yy = np.meshgrid(x, y)
    zz = np.empty((101, 101))
    for i in range(101):
        for j in range(101):
            val = np.array([x[i], y[j]])
            zz[i, j] = objective_function(val)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz)
    plt.show()
    plt.savefig("surface.png")
