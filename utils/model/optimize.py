import numpy as np
from numba import njit, prange
from utils.model.core import generate_initial_sol, get_mating_pool, population_constraint_violation, \
    population_objective_value, population_fitness, do_crossover, do_mutation, truncation_core
from utils.model.constraint_handle import constraint_violation
from param import x_prob, a, b_cts, b_int, m_prob, p_cts, p_int, size, max_iter, max_stall, max_run


@njit
def opt_single(objective_function, x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, nonlinear_functions,
               tol=1e-4):
    avg_fitness = np.empty(max_iter)
    best_fitness = np.empty(max_iter)
    pool = generate_initial_sol(x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, size)
    dim1, dim2 = pool.shape
    pool_mat = np.empty((dim1, dim2, max_iter))
    best_obj = np.inf
    best_ind = pool[0]
    stall = 0
    count = 0
    for i in range(max_iter):
        pool_mat[:, :, i] = pool.copy()
        violation = population_constraint_violation(pool, lin_lhs, lin_rhs, x_cts, x_int,
                                                    lb_cts, ub_cts, lb_int, ub_int, nonlinear_functions)
        obj_val = population_objective_value(pool, objective_function)
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
            print(f"Optimization terminated, number of iterations: {i}, max stall of {max_stall} reached")
            return best_obj, best_ind, avg_fitness[:count], best_fitness[:count], pool_mat[:, :, :count]
    print(f"Genetic Algorithm completed with maximum iteration {max_iter}")
    return best_obj, best_ind, avg_fitness[:count], best_fitness[:count], pool_mat[:, :, :count]


@njit(parallel=True)
def opt(objective_function, x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int, lin_lhs, lin_rhs, nonlinear_functions):
    num_vars = len(x_int) + len(x_cts)
    res = np.empty(max_run)
    ind = np.empty((max_run, num_vars))
    avg_fit_mat = np.empty((max_run, max_iter))
    best_fit_mat = np.empty((max_run, max_iter))
    pts_mat = np.empty(max_run)
    for i in prange(max_run):
        best_obj, best_ind, avg_fitness, best_fitness, _ = opt_single(objective_function, x_cts, x_int, lb_cts, ub_cts,
                                                                      lb_int, ub_int, lin_lhs, lin_rhs,
                                                                      nonlinear_functions)
        res[i] = best_obj
        ind[i] = best_ind
        num_pts = len(avg_fitness)
        avg_fit_mat[i, :num_pts] = avg_fitness
        best_fit_mat[i, :num_pts] = best_fitness
        pts_mat[i] = num_pts

    glb_best_idx = np.argmin(res)
    best_obj = res[glb_best_idx]
    best_ind = ind[glb_best_idx]
    avg_fit = avg_fit_mat[glb_best_idx, :pts_mat[glb_best_idx]]
    best_fit = best_fit_mat[glb_best_idx, :pts_mat[glb_best_idx]]
    # constraint violation
    violation = constraint_violation(best_ind.reshape(num_vars, 1), lin_lhs, lin_rhs, nonlinear_functions)
    return best_obj, best_ind, avg_fit, best_fit, violation
