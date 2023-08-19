import numpy as np
from numba import njit, prange
import math
import random
from utils.numba.random import randint, randint_unique


@njit
def generate_initial_sol(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, size):
    """
    :param x_cts: indices of continuous decision variables in the variable array
    :param x_int: indices of integer decision variables in the variable array
    :param lb_cts: lower bound of each continuous decision variable
    :param ub_cts: upper bound of each continuous decision variable
    :param lb_int: lower bound of each integer decision variable
    :param ub_int: upper bound of each integer decision variable
    :param size: number of initial individuals
    :return:
    """
    num_cts = len(lb_cts)
    num_int = len(lb_int)
    num_dv = num_cts + num_int
    chromosome = np.empty((size, num_dv))
    for j in range(num_cts):
        cts_individual = np.random.uniform(lb_cts[j], ub_cts[j], size)
        for k in range(size):
            chromosome[k, x_cts[j]] = cts_individual[k]
    for j in range(num_int):
        int_individual = randint(lb_int[j], ub_int[j] + 1, size)
        for k in range(size):
            chromosome[k, x_int[j]] = int_individual[k]
    return chromosome


@njit
def laplace_crossover(x1: np.array, x2: np.array, beta: np.array) -> tuple[np.array, np.array]:
    delta = np.abs(x1 - x2)
    inc = np.multiply(beta, delta)
    y1 = x1 + inc
    y2 = x2 + inc
    return y1, y2


@njit(fastmath=True)
def laplace_transform(a: int, b: float, u: np.array, r: np.array) -> np.ndarray:
    num = len(u)
    beta = np.empty(num)
    for i in range(num):
        if r[i] <= 0.5:
            beta[i] = a - b * math.log(u[i])
        else:
            beta[i] = a + b * math.log(u[i])
    return beta


@njit
def power_mutation(x, xl, xu, s, r):
    """
    power mutation is applied element wise
    :param x:
    :param xl:
    :param xu:
    :param s:
    :param r:
    :return:
    """
    if abs(xu - x) > 1e-4:
        t = (x - xl) / (xu - x)
    else:
        t = 1
    if t < r:
        return x - s * (x - xl)
    else:
        return x + s * (xu - x)


@njit
def mutation_prob(s1, p):
    return s1 ** p


@njit
def tournament_selection(tournament_size: int, candidates: np.array, fitness: np.array) -> np.array:
    n = len(candidates)
    sampled = randint_unique(0, n, tournament_size)
    sample_fitness = fitness[sampled]
    best_ind = np.argmin(sample_fitness)
    return candidates[sampled[best_ind]]


@njit(parallel=True)
def get_mating_pool(population, fitness, pool_size=0, tournament_size=3):
    # todo: performance enhancement
    if pool_size == 0:
        pool_size = len(population)
    res = np.empty((pool_size, len(population[0])))
    for i in prange(pool_size):
        selected = tournament_selection(tournament_size, population, fitness)
        for j in range(len(population[0])):
            res[i][j] = selected[j]
    return res

