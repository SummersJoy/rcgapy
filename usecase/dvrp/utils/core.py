import numpy as np
from numba import njit, int32, prange
from usecase.dvrp.utils.io.manipulate import fill_zero
from utils.numba.bisect import bisect
from usecase.dvrp.utils.heuristics.local_search.first_descend import descend, neighbourhood_gen
from usecase.dvrp.utils.split import split, label2route
from usecase.dvrp.utils.route.repr import decoding, get_trip_dmd, trip_lookup


@njit(fastmath=True)
def lox(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    TSP like crossover operator
    """
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


@njit(fastmath=True)
def check_spaced(space_hash: np.ndarray, val: float, delta: float) -> bool:
    """
    check if new chromosome is well-spaced in the population in O(1) time
    """
    idx = int32(val / delta)
    # n = len(space_hash)
    # if idx > n:
    #     space_hash = np.concatenate((space_hash, np.zeros(2 * idx - n, dtype=int32)))
    #     space_hash[idx] = 1
    #     return True
    if space_hash[idx]:
        return False
    else:
        space_hash[idx] = 1
        return True


@njit(fastmath=True)
def optimize(cx, cy, max_route_len, n, q, d, c, w, max_dist, size, pm, alpha, beta, delta, max_agl):
    best_vec = np.empty(alpha)
    avg_vec = np.empty(alpha)
    pool, ind_fit, restart = get_initial_solution(n, size, q, d, c, w, max_dist, delta)
    ordered_idx = np.argsort(ind_fit)
    pool = pool[ordered_idx, :]
    ind_fit = ind_fit[ordered_idx]
    neighbor = neighbourhood_gen(cx, cy, max_agl)
    space_hash = np.zeros(50000, dtype=int32)
    for sol in pool:
        _, fitness = split(n, sol, q, d, c, w, max_dist)
        hash_idx = int(fitness / delta)
        space_hash[hash_idx] = 1
    a = 0
    b = 0
    mid = size // 2
    while a != alpha and b != beta:
        p1 = binary_tournament_selection(pool)
        p2 = binary_tournament_selection(pool)
        child1, child2 = lox(p1[1:], p2[1:])
        child = child1 if np.random.random() < 0.5 else child2
        child = fill_zero(n, child)
        label, val = split(n, child, q, d, c, w, max_dist)
        trip = label2route(n, label, child, max_route_len)
        k = np.random.randint(mid, size)
        modified_fitness = np.concatenate((ind_fit[:k], ind_fit[k + 1:]))
        if np.random.random() < pm:
            trip_dmd = get_trip_dmd(trip, q)
            f = val
            lookup = trip_lookup(trip, n)
            mutation(trip, n, c, val, trip_dmd, q, w, lookup, neighbor)
            chromosome = decoding(trip, n)
            _, fitness = split(n, chromosome, q, d, c, w, max_dist)
            is_spaced = check_spaced(space_hash, fitness, delta)
            if is_spaced:
                child = chromosome
                val = fitness
            else:
                val = f
                is_spaced = check_spaced(space_hash, val, delta)
        else:
            is_spaced = check_spaced(space_hash, val, delta)
        if is_spaced:
            space_hash[int(ind_fit[k] / delta)] = 0  # remove hashed value from spack_hash
            best_vec[a] = ind_fit[0]
            # avg_vec[a] = np.mean(ind_fit)
            a += 1
            idx = bisect(modified_fitness, val) + 1
            if idx == k:
                pool[k] = child
                ind_fit[k] = val
            elif idx < k:
                pool = np.concatenate((pool[:idx, :], child.reshape((1, n + 1)), pool[idx:k, :], pool[(k + 1):, :]))
                ind_fit = np.concatenate((ind_fit[:idx], val * np.ones(1), ind_fit[idx:k], ind_fit[(k + 1):]))
            else:
                idx += 1
                pool = np.concatenate((pool[:k, :], pool[(k + 1):idx, :], child.reshape((1, n + 1)), pool[idx:, :]))
                ind_fit = np.concatenate((ind_fit[:k], ind_fit[(k + 1):idx], val * np.ones(1), ind_fit[idx:]))
            if idx == 0:  # incumbent solution found
                b = 0
            else:  # stall
                b += 1
    return pool, ind_fit, best_vec, avg_vec


@njit
def mutation(trip, n, c, fitness, trip_dmd, q, w, lookup, neighbor):
    prev = 0
    while True:
        gain = descend(trip, n, c, trip_dmd, q, w, lookup, neighbor)
        if gain < 0 or abs(gain - prev) < 1e-4:
            break
        else:
            fitness -= gain
            prev = gain
    return fitness


@njit(parallel=True)
def multi_start(cx, cy, max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta, delta, rho):
    fitness = np.empty(rho)
    sol = np.empty((rho, n + 1), dtype=int32)
    for i in prange(rho):
        pool, ind_fit, _, _ = optimize(cx, cy, max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta, delta)
        fitness[i] = ind_fit[0]
        sol[i] = pool[0]
    idx = np.argmin(fitness)
    return sol[idx], fitness[idx]


@njit
def find_best(cx, cy, max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta, delta, max_agl, best_sol):
    pool, ind_fit, _, _ = optimize(cx, cy, max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta,
                                   delta, max_agl)
    count = 0
    best = np.inf
    while ind_fit[0] > best_sol:
        print(count, ind_fit[0], best)
        pool, ind_fit, _, _ = optimize(cx, cy, max_route_len, n, q, d, c, w, max_load, size, pm, alpha, beta,
                                       delta, max_agl)
        count += 1
        if ind_fit[0] < best:
            best = ind_fit[0]
    return pool, ind_fit, count
