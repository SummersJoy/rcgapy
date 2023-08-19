import numpy as np
from numba import jit


def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print('No instructions found')


# basic SIMD
@jit(nopython=True)
def sqdiff(x, y):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = (x[i] - y[i]) ** 2
    return out


x32 = np.linspace(1, 2, 10000, dtype=np.float32)
y32 = np.linspace(2, 3, 10000, dtype=np.float32)
sqdiff(x32, y32)

x64 = x32.astype(np.float64)
y64 = y32.astype(np.float64)
sqdiff(x64, y64)
print(sqdiff.signatures)

# %timeit sqdiff(x32, y32)
# %timeit sqdiff(x64, y64)

print('float32:')
find_instr(sqdiff, keyword='subp', sig=0)
print('---\nfloat64:')
find_instr(sqdiff, keyword='subp', sig=1)