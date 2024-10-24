import numpy as np
import scipy as sp


def basis_functions(n):
    return lambda x: (1 - x ** 2) * sp.special.eval_jacobi(n, 1, 1, x)


def get_derivative(f, degree=1):
    return lambda x0: sp.misc.derivative(f, x0, n=degree, dx=1e-2)


def get_dot(a, b, f, g):
    return sp.integrate.quad(lambda x: f(x) * g(x), a, b)[0]


def galerkin_method(a, b, p, q, r, f, N):
    L = np.vectorize(
        lambda w:
        lambda x: p(x) * get_derivative(w, 2)(x) + q(x) * get_derivative(w)(x) + r(x) * w(x)
    )

    w = [basis_functions(i) for i in range(N)]
    Lw = L(w)

    matrix = np.zeros((N, N))
    result_vector = np.zeros((N, 1))

    for i in range(N):
        for j in range(N):
            matrix[i, j] = get_dot(a, b, Lw[j], w[i])
        result_vector[i] = get_dot(a, b, f, w[i])

    c_vector = np.linalg.solve(matrix, result_vector)

    return lambda x: sum(c_vector[i] * w[i](x) for i in range(N))
