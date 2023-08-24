import matplotlib.pyplot as plt


def ga_plot_single(avg, bst):
    npts = len(avg)
    x = list(range(npts))
    plt.plot(x, avg, ".")
    plt.plot(x, bst, "*")

    plt.xlabel("Generation")
    plt.ylabel("Fitness value")
    plt.title("Genetic Algorithm Numba")
    plt.show()
