import numpy as np

from usecase.dvrp.utils.io.read import read_xml
from usecase.dvrp.utils.io.manipulate import get_dist_mat, reformat_depot, fill_zero
from usecase.dvrp.utils.split import split, get_max_route_len
from usecase.dvrp.utils.test import trip_test
from usecase.dvrp.utils.heuristics.route_construction.sweep import sweep_constructor

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


def test_gm():
    gm = sweep_constructor(cx, cy, q, c, max_route_len, w, np.inf)
    trip_test(gm, n)

