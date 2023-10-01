import matplotlib.pyplot as plt


def base_plot(cx, cy):
    n = len(cx)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.scatter(cx, cy)
    ax.title.set_text(f"CVRP with {n - 1} customers")
    for i, txt in enumerate(range(n)):
        ax.annotate(txt, (cx[i], cy[i] + 0.01))
    return fig, ax


def plot_trip(cx, cy, t, ax):
    lst = []
    for i in t:
        if i:
            lst.append(i)
        else:
            break
    ax.plot(cx[lst], cy[lst])


def plot_sol(cx, cy, trip):
    fig, ax = base_plot(cx, cy)
    for t in trip:
        plot_trip(cx, cy, t, ax)
    plt.show()
