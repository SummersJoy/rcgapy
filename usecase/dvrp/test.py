import numpy as np

from usecase.dvrp.utils.io.read import read_xml
from usecase.dvrp.utils.io.manipulate import get_dist_mat, reformat_depot, fill_zero
from usecase.dvrp.utils.split import split, get_max_route_len, label2route
from usecase.dvrp.utils.test import trip_test, test_operation_m2
from usecase.dvrp.utils.heuristics.route_construction.sweep import sweep_constructor
from usecase.dvrp.utils.core import optimize

pm = 0.2
size = 30
max_load = 10000
alpha = 30000
beta = 10000
delta = 0.5
rho = 16  # number of restarts
max_agl = 45.  # angle threshold
filename = "/mnt/d/ga/ga/data/dvrp/christofides/CMT05.xml"
cx, cy, q, w, depot = read_xml(filename)
cx = reformat_depot(cx)
cy = reformat_depot(cy)
c = get_dist_mat(cx, cy)
n = len(cx) - 1
d = np.zeros(n)
q = fill_zero(n, q)
d = fill_zero(n, d)
max_route_len = get_max_route_len(q, w)

pool, ind_fit = optimize(cx, cy, max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta,
                                            delta, max_agl)


def test_gm():
    gm = sweep_constructor(cx, cy, q, c, max_route_len, w, np.inf)
    trip_test(gm, n)


def test_m2():
    s = pool[0]
    label, fitness = split(n, s, q, d, c, w, max_load)
    trip = label2route(n, label, s, max_route_len)

    test_operation_m2(c, trip, n, q, np.ones((len(trip), 10)))
