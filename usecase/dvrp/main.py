import numpy as np
from usecase.dvrp.utils.io.read import read_xml
from usecase.dvrp.utils.io.manipulate import get_dist_mat, reformat_depot, fill_zero
from usecase.dvrp.utils.core import optimize, multi_start, find_best
from usecase.dvrp.utils.split import split, get_max_route_len
from usecase.dvrp.utils.visualize.route import plot_sol
from usecase.dvrp.utils.split import label2route
from usecase.dvrp.utils.test import test_operation_m2
from usecase.dvrp.utils.heuristics.route_construction.sweep import sweep_constructor

# parameters
pm = 0.2
size = 30
max_dist = 10000
alpha = 30000
beta = 10000
delta = 0.5
rho = 16  # number of restarts
max_agl = 22.5  # angle threshold
filename = "/mnt/d/ga/ga/data/dvrp/christofides/CMT05.xml"

cx, cy, q, w, depot = read_xml(filename)
cx = reformat_depot(cx)
cy = reformat_depot(cy)
c = np.round(get_dist_mat(cx, cy), 3)
n = len(cx) - 1
d = np.zeros(n)
q = fill_zero(n, q)
d = fill_zero(n, d)
max_route_len = get_max_route_len(q, w)

# main loop
pool, ind_fit = optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl)
print(ind_fit[0])
# %timeit pool, ind_fit, best_vec, avg_vec = optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl)
# trip = sweep_constructor(cx, cy, q, c, max_route_len, w, max_dist)
# procedure of finding a solution within threshold
import time

start = time.perf_counter()
count_vec = []
for _ in range(10):
    pool, ind_fit, count = find_best(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta,
                                     max_agl, 1350)
    count_vec.append(count)
print(np.mean(count_vec))
print(time.perf_counter() - start)
# test
s = pool[0]
label, fitness = split(n, s, q, d, c, w, max_dist)
trip = label2route(n, label, s, max_route_len)

# test_operation_m2(c, trip, n, q, np.ones((len(trip), 10)))
# trip_test(trip, n)
# test_operation_m1(c, trip, n)
for i in range(size):
    _, f = split(n, pool[i], q, d, c, w, max_dist)
    if f != ind_fit[i]:
        print("failed")

plot_sol(cx, cy, trip)

# sol, fit = multi_start(cx, cy, max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta, delta, rho)
# label, fitness = split(n, sol, q, d, c, w, max_load)
# trip = label2route(n, label, s, max_route_len)
# plot_sol(cx, cy, trip)
