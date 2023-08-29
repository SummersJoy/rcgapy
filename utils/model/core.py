import numpy as np
from numba import njit, prange, boolean
from utils.model.constraint_handle import constraint_violation, bound_violation
from utils.model.operator import laplace_crossover, laplace_transform, power_mutation, mutation_prob, \
    tournament_selection
from utils.numba.random import randint


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
def generate_initial_sol_cts(x_cts, lb_cts, ub_cts, size):
    # pure continuous variable type
    num_cts = len(lb_cts)
    num_dv = num_cts
    chromosome = np.empty((size, num_dv))
    fill_cts(x_cts, num_cts, lb_cts, ub_cts, size, chromosome)
    return chromosome


@njit
def generate_initial_sol_int(x_int, lb_int, ub_int, size):
    # pure integer type
    num_int = len(lb_int)
    num_dv = num_int
    chromosome = np.empty((size, num_dv))
    fill_int(x_int, num_int, lb_int, ub_int, size, chromosome)
    return chromosome


@njit
def fill_cts(x_cts, num_cts, lb_cts, ub_cts, size, chromosome):
    for j in range(num_cts):
        cts_individual = np.random.uniform(lb_cts[j], ub_cts[j], size)
        for k in range(size):
            chromosome[k, x_cts[j]] = cts_individual[k]


@njit
def fill_int(x_int, num_int, lb_int, ub_int, size, chromosome):
    for j in range(num_int):
        int_individual = randint(lb_int[j], ub_int[j] + 1, size)
        for k in range(size):
            chromosome[k, x_int[j]] = int_individual[k]


@njit(parallel=True)
def population_constraint_violation(population, lin_lhs, lin_rhs, x_cts, x_int, lb_cts, ub_cts, lb_int,
                                    ub_int, nonlinear_functions):
    n, m = population.shape
    violation = np.empty(n)
    for i in prange(n):
        var = population[i]
        violation[i] = constraint_violation(var.reshape(m, 1), lin_lhs, lin_rhs, nonlinear_functions) + \
                       bound_violation(var, x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int)
    return violation


@njit(parallel=True)
def population_objective_value(population, objective_function):
    n, m = population.shape
    obj_val = np.empty(n)
    for i in prange(n):
        var = population[i]
        obj_val[i] = objective_function(var)
    return obj_val


@njit
def population_fitness(violation, obj_val, tol=1e-4):
    n = len(violation)
    is_feasible = np.zeros(n, dtype=boolean)
    fitness = np.empty(n)
    worst_obj = 0.
    for i in range(n):
        violation_i = violation[i]
        if violation_i <= tol:
            is_feasible[i] = True
            obj_val_i = obj_val[i]
            fitness[i] = obj_val_i
            if obj_val_i >= worst_obj:
                worst_obj = obj_val_i
        else:
            fitness[i] = violation_i
    fitness[~is_feasible] += worst_obj
    return fitness


@njit(parallel=True)
def do_crossover(mating_pool, x_prob, a, b_cts, b_int, x_cts, x_int):
    n, m = mating_pool.shape
    num_parents = n // 2
    num_cts = len(x_cts)
    num_int = len(x_int)
    rand = np.random.random(num_parents)
    u_cts = np.random.random((num_parents, num_cts))
    r_cts = np.random.random((num_parents, num_cts))
    u_int = np.random.random((num_parents, num_int))
    r_int = np.random.random((num_parents, num_int))
    for i in prange(num_parents):
        if rand[i] <= x_prob:
            x1 = mating_pool[2 * i]
            x2 = mating_pool[2 * i + 1]
            y1 = np.empty(m)  # initialize y1
            y2 = np.empty(m)  # initialize y2
            beta_cts = laplace_transform(a, b_cts, u_cts[i], r_cts[i])
            y1_cts, y2_cts = laplace_crossover(x1[x_cts], x2[x_cts], beta_cts)
            y1[x_cts] = y1_cts
            y2[x_cts] = y2_cts
            beta_int = laplace_transform(a, b_int, u_int[i], r_int[i])
            y1_int, y2_int = laplace_crossover(x1[x_int], x2[x_int], beta_int)
            y1[x_int] = y1_int
            y2[x_int] = y2_int
            for j in range(m):
                mating_pool[2 * i, j] = y1[j]
                mating_pool[2 * i + 1, j] = y2[j]
    return mating_pool


@njit
def do_mutation(mating_pool, m_prob, p_cts, p_int, x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int):
    mutate_core(mating_pool, x_cts, m_prob, p_cts, lb_cts, ub_cts)
    mutate_core(mating_pool, x_int, m_prob, p_int, lb_int, ub_int)


@njit
def mutate_core(mating_pool, x_cts, m_prob, p_cts, lb_cts, ub_cts):
    n, m = mating_pool.shape
    num_cts = len(x_cts)
    rand_cts = np.random.random((n, num_cts))
    power_distribution = np.random.random((n, num_cts))
    r = np.random.random((n, num_cts))
    for i in range(n):
        for j in range(num_cts):
            if rand_cts[i, j] <= m_prob:
                s = mutation_prob(power_distribution[i, j], p_cts)
                mating_pool[i, x_cts[j]] = power_mutation(mating_pool[i, x_cts[j]], lb_cts[j], ub_cts[j], s, r[i, j])


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


@njit
def truncation(val, rng, tol=1e-4):
    ceil = np.ceil(val)
    floor = np.floor(val)
    if ceil - val <= tol:
        return val
    if val - floor <= tol:
        return val
    if rng < 0.5:
        return ceil
    return floor


@njit
def truncation_core(sol, x_int):
    n = len(sol)
    num_int = len(x_int)
    rng = np.random.random((n, num_int))
    for i in range(n):
        for j in range(num_int):
            sol[i, x_int[j]] = truncation(sol[i, x_int[j]], rng[i, j])
