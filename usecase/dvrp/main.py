import numpy as np
from usecase.dvrp.utils.io.read import read_xml
from usecase.dvrp.utils.io.manipulate import get_dist_mat
from usecase.dvrp.utils.core import get_initial_solution

filename = "D:\\ga\\ga\\data\\dvrp\\christofides\\CMT01.xml"
cx, cy, q, w = read_xml(filename)
c = get_dist_mat(cx, cy)
n = len(cx) - 1
size = 100
d = np.zeros(n)
max_load = 100
delta = 1
pool, ind_fit, restart = get_initial_solution(n, size, q, d, c, w, max_load, delta)