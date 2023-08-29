from numba import njit
# from utils.model.expr import nonlinear_functions


@njit()
def nonlinear_constraints(x, nonlinear_functions):
    vals = nonlinear_functions(x)
    if len(vals) == 1:
        return vals[0]
    res = 0.
    for val in vals:
        if val > 0.:
            res += val
    return res


@njit
def linear_constraint_violation(lhs, rhs, var):
    diff = lhs.dot(var) - rhs
    res = 0.
    diff = diff.flatten()
    for i in range(len(diff)):
        val = diff[i]
        if val > 0:
            res += val
    return res


@njit
def constraint_violation(var, lin_lhs, lin_rhs, nonlinear_functions):
    lin_vio = linear_constraint_violation(lin_lhs, lin_rhs, var)  # compute linear inequality violation
    non_vio = nonlinear_constraints(var.T[0], nonlinear_functions)  # compute non-linear inequality violation
    return lin_vio + non_vio


@njit
def bound_violation(var, x_cts, x_int, lb_cts, ub_cts, lb_int, ub_int):
    var_cts = var[x_cts]
    var_int = var[x_int]
    cts_bound_violation = bound_violation_core(var_cts, lb_cts, ub_cts)
    int_bound_violation = bound_violation_core(var_int, lb_int, ub_int)
    return cts_bound_violation + int_bound_violation


@njit
def bound_violation_core(var_x, lb, ub):
    n = len(var_x)
    res = 0.
    for i in range(n):
        val = var_x[i]
        if val < lb[i]:
            res += lb[i] - val
        if val > ub[i]:
            res += val - ub[i]
    return res

# arg = np.array([[1.2, 1.8, 2.5, 1., 1., 1., 1.]]).T
# rhs_lin = np.array([[5., 1.2, 1.8, 2.5, 1.2]]).T
# linear_ineq = np.array([
#     [1., 1., 1., 1., 1., 1., 0.],
#     [1., 0., 0., 1., 0., 0., 0.],
#     [0., 1., 0., 0., 1., 0., 0.],
#     [0., 0., 1., 0., 0., 1., 0.],
#     [1., 0., 0., 0., 0., 0., 1.]
# ])
# rhs_nonlinear = np.array([5.5, 1.64, 4.25, 4.64])
#
# lin_vio = linear_constraint_violation(linear_ineq, rhs_lin, arg)
# # %timeit nonlinear_const_violation(rhs_nonlinear, arg.flatten())
# nonlinear_constraint1(arg)
# single_constraint_violation(nonlinear_constraint1, rhs_nonlinear[0], arg)
# nonlinear_const_violation(rhs_nonlinear, arg)
#
# vio = constraint_violation(arg, linear_ineq, rhs_lin, rhs_nonlinear)
#
# obj_val = objective_function(arg)
