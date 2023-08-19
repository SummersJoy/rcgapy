import numpy as np
from numba import njit, prange
from utils.model.operator import generate_initial_sol, get_mating_pool
from utils.model.core import population_constraint_violation, population_objective_value, population_fitness, \
    do_crossover, do_mutation, truncation_core


@njit
def _opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, non_rhs, x_prob, a, b_cts, b_int, m_prob,
         p_cts,
         p_int, size, max_iter, max_stall):
    pool = generate_initial_sol(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, size)
    best_obj = np.inf
    best_ind = pool[0]
    stall = 0
    for i in range(max_iter):
        violation = population_constraint_violation(pool, lin_lhs, lin_rhs, non_rhs, x_cts, x_int,
                                                    lb_cts, ub_cts, lb_int, ub_int)
        obj_val = population_objective_value(pool)
        fitness = population_fitness(violation, obj_val)
        idx = np.argmin(fitness)
        if fitness[idx] <= best_obj:
            best_obj = fitness[idx]
            best_ind = pool[idx]
            stall = 0
        else:
            stall += 1
        pool = get_mating_pool(pool, fitness)
        pool = do_crossover(pool, x_prob, a, b_cts, b_int, x_cts, x_int)
        do_mutation(pool, m_prob, p_cts, p_int, x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int)
        truncation_core(pool, x_int)
        if stall == max_stall:
            print(f"Genetic Algorithm quited at iteration {i}, max stall of {max_stall} reached")
            return best_obj, best_ind
    return best_obj, best_ind


@njit(parallel=True)
def opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, non_rhs, x_prob, a, b_cts, b_int, m_prob, p_cts,
        p_int, size, max_iter, max_stall, max_run):
    res = np.empty(max_run)
    ind = np.empty((max_run, len(x_int) + len(x_cts)))
    for i in prange(max_run):
        best_obj, best_ind = _opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, non_rhs, x_prob, a,
                                  b_cts, b_int, m_prob, p_cts, p_int, size, max_iter, max_stall)
        res[i] = best_obj
        ind[i] = best_ind

    glb_best_idx = np.argmin(res)
    return res[glb_best_idx], ind[glb_best_idx]
