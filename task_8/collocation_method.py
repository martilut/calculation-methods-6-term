import numpy as np
import scipy.linalg as la


def collocation_method(p, q, f, a, b, alpha, beta, N):

    collocation_points = np.linspace(a, b, N)

    A = np.zeros((N, N))
    B = np.zeros(N)

    for i, x_i in enumerate(collocation_points):
        basis = basis_functions(x_i, N)
        dbasis = basis_derivatives(x_i, N)
        d2basis = basis_second_derivatives(x_i, N)

        A[i, :] = d2basis + p(x_i) * dbasis + q(x_i) * basis
        B[i] = f(x_i)

    A[0, :] = basis_functions(a, N)
    B[0] = alpha

    A[-1, :] = basis_functions(b, N)
    B[-1] = beta

    coefficients = la.solve(A, B)

    def approximate_solution(x):
        basis = basis_functions(x, N)
        return np.dot(coefficients, basis)

    return approximate_solution












def basis_functions(x, n):
    return np.array([x ** i for i in range(n)])

def basis_derivatives(x, n):
    return np.array([i * x ** (i - 1) if i > 0 else 0 for i in range(n)])

def basis_second_derivatives(x, n):
    return np.array([i * (i - 1) * x ** (i - 2) if i > 1 else 0 for i in range(n)])