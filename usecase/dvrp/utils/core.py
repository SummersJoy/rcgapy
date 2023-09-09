import numpy as np
from numba import njit, int32
from usecase.dvrp.utils.split import split


@njit
def lox(p1, p2):
    n = len(p1)
    pos1 = np.random.randint(0, n)
    pos2 = np.random.randint(0, n)
    c1 = np.empty(n, dtype=int32)
    c2 = np.empty(n, dtype=int32)
    while pos2 == pos1:
        pos2 = np.random.randint(0, n)
    i = min(pos1, pos2)
    j = max(pos1, pos2)
    c1[i:j] = p1[i:j]
    c2[i:j] = p2[i:j]
    fill_chromosome(p1, p2, c1, i, j, n)
    fill_chromosome(p2, p1, c2, i, j, n)
    return c1, c2, i, j


@njit
def fill_chromosome(p1, p2, c1, i, j, n):
    count = 0
    for t in p2[j:]:
        if j + count < n:
            if t not in p1[i:j]:
                c1[j + count] = t
                count += 1
        else:
            break
    count_f = 0
    for t in p2[:j]:
        if t not in p1[i:j]:
            if j + count < n:
                c1[j + count] = t
                count += 1
            else:
                c1[count_f] = t
                count_f += 1


# p1 = np.random.permutation(50) + 1
# p2 = np.random.permutation(50) + 1
# # i = 4
# # j = 6
# %timeit c1, c2, pos1, pos2 = lox(p1, p2)
# assert len(c1) == 50
# assert len(c2) == 50

@njit
def get_new_ind(n):
    tmp = np.random.permutation(n) + 1
    s = np.empty(n + 1, dtype=int32)
    s[0] = 0
    s[1:] = tmp
    return s


@njit
def get_initial_solution(n, size, q, d, c, w, max_load, delta):
    res = np.empty((size, n + 1), dtype=int32)
    ind_fitness = np.empty(size)
    restart = 0
    for i in range(size):
        s = get_new_ind(n)
        label, fitness = split(n, s, q, d, c, w, max_load)
        while True:
            well_spaced = True
            for j in range(i):
                if abs(fitness - ind_fitness[j]) < delta:
                    well_spaced = False
                    break
            if well_spaced:
                ind_fitness[i] = fitness
                break
            else:
                restart += 1
                s = get_new_ind(n)
                label, fitness = split(n, s, q, d, c, w, max_load)
        res[i] = s

    return res, ind_fitness, restart


# res = get_initial_solution(50, 100)
@njit
def binary_tournament_selection():
    pass
