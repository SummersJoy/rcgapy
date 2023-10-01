from numpy cimport ndarray
from numpy import empty, size

cdef extern:
    void c_abc_model_fortran(int n, double a, double b, double c, double *inflow, double *outflow)

def abc_model_fortran(double a, double b, double c, ndarray[double, mode="c"] inflow):
    cdef int N = size(inflow)
    cdef ndarray[double, mode="c"] outflow = empty(N, dtype="double")
    c_abc_model_fortran(N, a, b, c, &inflow[0], &outflow[0])
    return outflow
