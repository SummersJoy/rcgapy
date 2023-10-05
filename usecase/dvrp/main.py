import numpy as np
from usecase.dvrp.utils.io.read import read_xml
from usecase.dvrp.utils.io.manipulate import get_dist_mat, reformat_depot, fill_zero
from usecase.dvrp.utils.core import optimize, multi_start, find_best, get_new_ind
from usecase.dvrp.utils.split import split, get_max_route_len, label2route
from usecase.dvrp.utils.visualize.route import plot_sol
from usecase.dvrp.utils.test import test_operation_m2
from usecase.dvrp.utils.heuristics.route_construction.sweep import sweep_constructor

# parameters
pm = 0.1
size = 30
max_dist = 10000
alpha = 30000
beta = 10000
delta = 0.5
rho = 16  # number of restarts
max_agl = 45.  # angle threshold
filename = "/mnt/d/ga/ga/data/dvrp/christofides/CMT05.xml"
filename = "/mnt/d/ga/ga/data/dvrp/golden/Golden_02.xml"
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
heuristic_sol = np.empty((3, n + 1), dtype=int)
for i in range(3):
    ind = get_new_ind(n)
    heuristic_sol[i] = ind
h_sol = heuristic_sol
pool, ind_fit = optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl,
                         heuristic_sol)
print(ind_fit[0])
best_val = np.inf
best_sol = None
for i in range(1000):
    if i == 0:
        heuristic_sol[0] = pool[0]
    if i == 1:
        heuristic_sol[1] = pool[1]
    if i == 2:
        heuristic_sol[2] = pool[2]
    pool, ind_fit = optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl,
                             heuristic_sol)
    _, val = split(n, heuristic_sol[0], q, d, c, w, max_dist)
    if ind_fit[0] < val:
        heuristic_sol[0] = pool[0]
    _, val = split(n, heuristic_sol[1], q, d, c, w, max_dist)
    if ind_fit[1] < val:
        heuristic_sol[1] = pool[1]
    _, val = split(n, heuristic_sol[2], q, d, c, w, max_dist)
    if ind_fit[2] < val:
        heuristic_sol[2] = pool[2]
    if ind_fit[0] < best_val:
        best_val = ind_fit[0]
        best_sol = pool[0]
        print(f"Current best: {np.round(best_val, 3)}")
    else:
        print(f"Not improved in iteration {i}, current best:{np.round(best_val, 3)}")

# %timeit pool, ind_fit, best_vec, avg_vec = optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl)
# trip = sweep_constructor(cx, cy, q, c, max_route_len, w, max_dist)
# procedure of finding a solution within threshold
import time

start = time.perf_counter()
count_vec = []
for _ in range(10):
    pool, ind_fit, count = find_best(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta,
                                     max_agl, 527)
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
space_hash = np.zeros(100000)
for i in range(size):
    _, f = split(n, pool[i], q, d, c, w, max_dist)
    hash_val = int(f / delta)
    if space_hash[hash_val]:
        raise ValueError("Solution is not spaced!")
    else:
        space_hash[hash_val] = 1.
    if f != ind_fit[i]:
        print("failed")

plot_sol(cx, cy, trip)
best_val = np.inf
best_sol = None
while True:
    sol, fit = multi_start(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, rho, max_agl,
                           heuristic_sol)
    print(fit)
    if fit < best_val:
        best_val = fit
        best_sol = sol
        print(f"Incumbent solution: {np.round(best_val, 3)}")
    else:
        print(f"Current best: {np.round(best_val, 3)}")
    if best_val <= 1300.:
        break
# label, fitness = split(n, sol, q, d, c, w, max_load)
# trip = label2route(n, label, s, max_route_len)
# plot_sol(cx, cy, trip)
