from utils.model.operator import laplace_crossover, laplace_transform

import numpy as np

a = 0
b = 0.15
num = 1000
x1 = np.random.random(1000)
x2 = np.random.random(1000)

u = np.random.random(1000)
r = np.random.random(1000)

beta = laplace_transform(a, b, u, r)
y1, y2 = laplace_crossover(x1, x2, beta)


def laplace_transform_py(a: int, b: float, u: np.array, r: np.array) -> np.ndarray:
    n = len(u)
    beta = np.empty(n)
    for i in range(n):
        if r[i] <= 0.5:
            beta[i] = a - b * np.log(u[i])
        else:
            beta[i] = a + b * np.log(u[i])
    return beta

# %timeit beta = laplace_transform(a, b, u, r)  # llvm compiled python code
# %timeit beta = laplace_transform_py(a, b, u, r) # naive python interpreted code


