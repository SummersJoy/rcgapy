import numpy as np
from usecase.dvrp.utils.io.read import read_xml
from usecase.dvrp.utils.io.manipulate import get_dist_mat, reformat_depot, fill_zero
from usecase.dvrp.utils.core import optimize
from usecase.dvrp.utils.split import split, get_max_route_len
from usecase.dvrp.utils.visualize.route import plot_sol
from usecase.dvrp.utils.split import label2route

# parameters
pm = 0.05
size = 30
max_load = 10000
alpha = 30000
beta = 2000
delta = 0.5
pho = 8  # number of restarts

filename = "D:\\ga\\ga\\data\\dvrp\\christofides\\CMT01.xml"
cx, cy, q, w, depot = read_xml(filename)
cx = reformat_depot(cx)
cy = reformat_depot(cy)
c = get_dist_mat(cx, cy)
n = len(cx) - 1
d = np.zeros(n)
q = fill_zero(n, q)
d = fill_zero(n, d)
max_route_len = get_max_route_len(q, w)

# main loop
pool, ind_fit = optimize(max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta, delta)

print(min(ind_fit))

# test
s = pool[0]
label, fitness = split(n, s, q, d, c, w, max_load)
trip = label2route(n, label, s, max_route_len)
# trip_test(trip, n)
# test_operation_m1(c, trip, n)
for i in range(size):
    _, f = split(n, pool[i], q, d, c, w, max_load)
    if f != ind_fit[i]:
        print("failed")

plot_sol(cx, cy, trip)
