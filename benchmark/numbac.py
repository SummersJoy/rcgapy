import time
import numpy as np
from cffi import FFI
from numpy.testing import assert_almost_equal
import gc

ffi = FFI()
ffi.cdef('void det_by_lu(double *y, double *B, int N);')
C = ffi.dlopen("/home/shiyao/numbabench/lu.dll")
c_det_by_lu = C.det_by_lu


def run_c(A, B, y, N):
    # run c code
    # B = numpy.zeros((N,N), order='F')
    # B[:,:] = A
    np.copyto(B, A)
    c_det_by_lu(ffi.cast("double *", y.ctypes.data),
                ffi.cast("double *", B.ctypes.data),
                ffi.cast("int", N))

    # check that result is correct
    L = np.tril(B, -1) + np.eye(N)
    U = np.triu(B)
    assert_almost_equal(L.dot(U), A)

    gc.disable()
    st = time.time()

    loops = 1 + min(1000000 // (N * N), 20000)

    for l in range(loops):
        np.copyto(B, A)
        c_det_by_lu(ffi.cast("double *", y.ctypes.data),
                    ffi.cast("double *", B.ctypes.data),
                    ffi.cast("int", N))

    et = time.time()
    gc.enable()

    return (et - st) / loops
