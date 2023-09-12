import numpy as np
from numba import njit, int32, jit
from usecase.dvrp.utils.split import split
from usecase.dvrp.utils.io.manipulate import fill_zero
from utils.numba.bisect import bisect
from usecase.dvrp.utils.heuristics.local_search.first_descend import descend
from usecase.dvrp.utils.split import split, label2route
from usecase.dvrp.utils.route.repr import decoding, get_trip_dmd


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
    return c1, c2


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
def binary_tournament_selection(population: np.ndarray) -> np.ndarray:
    """
    Tournament selection on a sorted population
    """
    n = len(population)
    id1 = np.random.randint(0, n)
    id2 = np.random.randint(0, n)
    while id1 == id2:
        id2 = np.random.randint(0, n)
    if id1 < id2:
        return population[id1]
    else:
        return population[id2]


@njit
def check_spaced(fitness: np.ndarray, val: float, delta: float) -> bool:
    """
    check if new chromosome is well-spaced in the population
    """
    for f in fitness:
        if abs(f - val) < delta:
            return False
    return True


# @njit
def optimize(pool, ind_fit, max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta):
    for i in range(alpha):
        p1 = binary_tournament_selection(pool)
        p2 = binary_tournament_selection(pool)
        child1, child2 = lox(p1[1:], p2[1:])
        child = child1 if np.random.random() < 0.5 else child2
        child = fill_zero(n, child)
        label, val = split(n, child, q, d, c, w, max_load)
        trip = label2route(n, label, child, max_route_len)
        k = np.random.randint(size // 2, size)
        modified_fitness = np.concatenate((ind_fit[:k], ind_fit[k + 1:]))
        if np.random.random() < pm:
            trip_dmd = get_trip_dmd(trip, q)
            f = val
            mutation(trip, n, c, val, trip_dmd, q, w)
            chromosome = decoding(trip, n)
            _, fitness = split(n, chromosome, q, d, c, w, max_load)
            is_spaced = check_spaced(modified_fitness, fitness, delta=1)
            if is_spaced:
                child = chromosome
                val = fitness
            else:
                val = f

        is_spaced = check_spaced(modified_fitness, val, delta=1)
        if is_spaced:
            idx = bisect(modified_fitness, val) + 1
            if idx == k:
                pool[k] = child
                ind_fit[k] = val
            elif idx < k:
                pool = np.concatenate((pool[:idx, :], child.reshape((1, n + 1)), pool[idx:k, :], pool[(k + 1):, :]))
                if len(pool) != size:
                    print(f"false 1: {len(pool)}, iter: {i}, idx: {idx}, k: {k}")
                ind_fit = np.concatenate((ind_fit[:idx], val * np.ones(1), ind_fit[idx:k], ind_fit[(k + 1):]))
            else:
                pool = np.concatenate((pool[:k, :], pool[(k + 1):idx, :], child.reshape((1, n + 1)), pool[idx:, :]))
                if len(pool) != size:
                    print(f"false 2: {len(pool)}, iter {i}, idx: {idx}, k: {k}")
                ind_fit = np.concatenate((ind_fit[:k], ind_fit[(k + 1):idx], val * np.ones(1), ind_fit[idx:]))
            # pool = np.concatenate((pool[:idx + 1, :], child.reshape((1, n + 1)), pool[idx + 2:, :]), )
            # ind_fit = np.concatenate((ind_fit[:idx + 1], val * np.ones(1), ind_fit[idx + 2:]))
    return pool, ind_fit


@njit
def mutation(trip, n, c, fitness, trip_dmd, q, w):
    prev = 0
    while True:
        gain = descend(trip, n, c, trip_dmd, q, w)
        if gain < 0 or abs(gain - prev) < 1e-4:
            break
        else:
            fitness -= gain
            prev = gain
        # print(f"gain: {gain}, fitness: {fitness}")
    return fitness
