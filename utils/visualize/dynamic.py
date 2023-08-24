from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use("TkAgg")  # On macOS


# todo: add init for blit
def ga_dynamic_single(avg_fit, best_fit, max_frame=100):
    l1 = avg_fit if len(avg_fit) < max_frame else avg_fit[:max_frame]
    l2 = best_fit if len(best_fit) < max_frame else best_fit[:max_frame]

    myvar = count(0, 1)

    # subplots() function you can draw
    # multiple plots in one figure
    fig, axes = plt.subplots()

    # set limit for x and y axis
    y_lb = min(min(l1), min(l2)) - 1
    y_ub = max(max(l1), max(l2)) + 1
    axes.set_ylim(y_lb, y_ub)
    axes.set_xlim(0, len(l1))
    axes.set_xlabel("Generations")
    axes.set_ylabel("Fitness value")

    # style for plotting line
    # plt.style.use("ggplot")

    # create 5 list to get store element
    # after every iteration
    x1, y1, y2, = [], [], []

    # set ani variable to call the
    # function recursively

    def animate(i):
        x1.append(next(myvar))
        y1.append((l1[i]))
        y2.append((l2[i]))

        axes.plot(x1, y1, "*", color="red")
        axes.plot(x1, y2, ".", color="blue")
        axes.legend(["Average Fitness", "Best Fitness"])
        axes.title.set_text(f"Best: {np.round(l2[i], 3)}, Average: {np.round(l1[i], 3)}")

    anim = FuncAnimation(fig, animate, interval=10, frames=len(l1), repeat=False)
    plt.show()
    return anim
