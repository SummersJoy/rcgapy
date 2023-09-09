import numpy as np
from usecase.dvrp.utils.io.read import read_xml
from usecase.dvrp.utils.io.manipulate import get_dist_mat, fill_zero, reformat_depot
from usecase.dvrp.utils.core import get_initial_solution
from usecase.dvrp.utils.split import split, label2route, get_max_route_len
from usecase.dvrp.utils.test import trip_test, test_operation_m1

filename = "D:\\ga\\ga\\data\\dvrp\\christofides\\CMT05.xml"
cx, cy, q, w, depot = read_xml(filename)
cx = reformat_depot(cx)
cy = reformat_depot(cy)
c = get_dist_mat(cx, cy)
n = len(cx) - 1
size = 100
d = np.zeros(n)
max_load = 10000
delta = 1
q = fill_zero(n, q)
d = fill_zero(n, d)
max_route_len = get_max_route_len(q, w)
pool, ind_fit, restart = get_initial_solution(n, size, q, d, c, w, max_load, delta)
s = pool[0]
label, fitness = split(n, s, q, d, c, w, max_load)
trip = label2route(n, label, s, max_route_len)
trip_test(trip, n)
test_operation_m1(c, trip, n)
