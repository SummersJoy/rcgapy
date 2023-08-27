import numpy as np
from numba import njit
import math
from utils.numba.random import randint_unique


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

