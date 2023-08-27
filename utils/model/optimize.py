import numpy as np
from numba import njit, prange
from utils.model.core import generate_initial_sol, get_mating_pool, population_constraint_violation, \
    population_objective_value, population_fitness, do_crossover, do_mutation, truncation_core


@njit
def _opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, x_prob, a, b_cts, b_int, m_prob,
         p_cts, p_int, size, max_iter, max_stall, tol=1e-4):
    avg_fitness = np.empty(max_iter)
    best_fitness = np.empty(max_iter)
    pool = generate_initial_sol(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, size)
    best_obj = np.inf
    best_ind = pool[0]
    stall = 0
    count = 0
    for i in range(max_iter):
        violation = population_constraint_violation(pool, lin_lhs, lin_rhs, x_cts, x_int,
                                                    lb_cts, ub_cts, lb_int, ub_int)
        obj_val = population_objective_value(pool)
        fitness = population_fitness(violation, obj_val)
        avg_fitness[i] = fitness.mean()
        idx = np.argmin(fitness)
        if fitness[idx] <= best_obj - tol:
            best_obj = fitness[idx]
            best_ind = pool[idx]
            stall = 0
            best_fitness[i] = fitness[idx]
        else:
            stall += 1
            best_fitness[i] = best_fitness[i - 1]
        pool = get_mating_pool(pool, fitness)
        pool = do_crossover(pool, x_prob, a, b_cts, b_int, x_cts, x_int)
        do_mutation(pool, m_prob, p_cts, p_int, x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int)
        truncation_core(pool, x_int)
        count += 1
        if stall == max_stall:
            print(f"Genetic Algorithm quited at iteration {i}, max stall of {max_stall} reached")
            return best_obj, best_ind, avg_fitness[:count], best_fitness[:count]
    print(f"Genetic Algorithm completed with maximum iteration {max_iter}")
    return best_obj, best_ind, avg_fitness[:count], best_fitness[:count]


@njit(parallel=True)
def opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, x_prob, a, b_cts, b_int, m_prob, p_cts,
        p_int, size, max_iter, max_stall, max_run):
    res = np.empty(max_run)
    ind = np.empty((max_run, len(x_int) + len(x_cts)))
    avg_fit_mat = np.empty((max_run, max_iter))
    best_fit_mat = np.empty((max_run, max_iter))
    pts_mat = np.empty(max_run)
    for i in prange(max_run):
        best_obj, best_ind, avg_fitness, best_fitness = _opt(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs,
                                                             lin_rhs, x_prob, a, b_cts, b_int, m_prob, p_cts,
                                                             p_int, size, max_iter, max_stall)
        res[i] = best_obj
        ind[i] = best_ind
        num_pts = len(avg_fitness)
        avg_fit_mat[i, :num_pts] = avg_fitness
        best_fit_mat[i, :num_pts] = best_fitness
        pts_mat[i] = num_pts

    glb_best_idx = np.argmin(res)
    return res[glb_best_idx], ind[glb_best_idx], avg_fit_mat[glb_best_idx, :pts_mat[glb_best_idx]], \
           best_fit_mat[glb_best_idx, :pts_mat[glb_best_idx]]
