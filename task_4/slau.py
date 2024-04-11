import numpy as np


def solve_for_diagonal(n, A, b):
    c = b.copy()
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                B[i, j] = 0
            else:
                B[i, j] = -A[i, j] / A[i, i]
        c[i] = c[i] / A[i, i]
    return B, c


def solve_for_positive(n, A, b):
    eigen_values = np.linalg.eig(A)[0]
    alpha = 2 / (min(eigen_values) + max(eigen_values))
    return np.eye(n) - alpha * A, alpha * b


def check_diagonal(A):
    non_diagonal_sum = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j:
                non_diagonal_sum += abs(A[i, j])
    for element in A.diagonal():
        if abs(element) < non_diagonal_sum:
            return False
    return True


def simple_iteration_method(A, b, eps):
    n = A.shape[0]
    is_positive = (np.allclose(A, np.transpose(A)) and min(np.linalg.eig(A)[0]) >= 0)
    is_diagonal = check_diagonal(A)  # матрица с диагональным преобладанием

    if is_diagonal:
        B, c = solve_for_diagonal(n, A, b)
    elif is_positive:
        B, c = solve_for_positive(n, A, b)
    else:
        raise RuntimeError('Matrix is not diagonal or positive')

    if max(np.linalg.eig(B)[0]) > np.linalg.norm(B):
        raise RuntimeError('Matrix B is not convergent')

    X, iterations_number = perform_iterations(B, c, eps)
    return X, iterations_number


def seidel_method(A, b, eps):
    D = np.triu(np.tril(A))
    R = np.triu(A) - D
    L = np.tril(A) - D
    inv_D_L = np.linalg.inv(D + L)

    B = np.matmul(-inv_D_L, R)
    c = np.matmul(inv_D_L, b)

    return perform_iterations(B, c, eps)


def perform_iterations(B, c, eps):
    norm_B = np.linalg.norm(B)
    current_X = c.copy()
    diff = None
    iterations_number = 0

    while diff is None or abs(diff) >= eps:
        previous_X = current_X
        current_X = np.dot(B, current_X) + c
        diff = norm_B * np.linalg.norm(current_X - previous_X) / (1 - norm_B)
        iterations_number += 1

    return current_X, iterations_number
