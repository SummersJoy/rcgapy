from usecase.dvrp.utils.heuristics.route_construction.sweep import sweep_constructor, sweep_optimizer
import numpy as np
from usecase.dvrp.utils.io.read import read_xml
from usecase.dvrp.utils.io.manipulate import get_dist_mat, reformat_depot, fill_zero

from usecase.dvrp.utils.split import split, get_max_route_len
from usecase.dvrp.utils.route.repr import get_route_len, get_trip_len

# parameters
pm = 0.2
size = 90
max_dist = 10000
alpha = 30000
beta = 2000
delta = 0.5
rho = 16  # number of restarts
max_agl = 45.  # angle threshold
filename = "/mnt/d/ga/ga/data/dvrp/christofides/CMT01.xml"
cx, cy, q, w, depot = read_xml(filename)
cx = reformat_depot(cx)
cy = reformat_depot(cy)
c = get_dist_mat(cx, cy)
n = len(cx) - 1
d = np.zeros(n)
q = fill_zero(n, q)
d = fill_zero(n, d)
max_route_len = get_max_route_len(q, w)
res = sweep_constructor(cx, cy, q, c, max_route_len, w, max_dist)

gm = np.zeros((5, 18))
r, b = 0, 0
for i in range(5):
    t, d, count = sweep_optimizer(res[i], cx, cy, c)
    route_len = get_route_len(c, t[1:])
    r += d
    b += route_len
    gm[i, :count] = t[1:]
