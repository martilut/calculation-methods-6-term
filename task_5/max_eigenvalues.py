import math
import numpy as np


def method_scalar(A, eps):
    previous_X = previous_eigenvalue = current_eigenvalue = None
    iterations = 0
    X = np.ones(A.shape[0])
    Y = np.ones(A.shape[0])

    while True:
        if previous_eigenvalue is not None \
                and previous_X is not None \
                and (current_eigenvalue - previous_eigenvalue) < eps:
            break
        if previous_X is not None and current_eigenvalue is not None:
            previous_eigenvalue = (np.matmul(X, Y)) / (np.matmul(previous_X, Y)) # (X^(k+1), Y^(k+1)) / (X^(k), Y^(k+1))
        previous_X = X
        X = np.matmul(A, X)  # X^(k+1) = A * X^(k)
        current_eigenvalue = (np.matmul(X, Y)) / (np.matmul(previous_X, Y))
        Y = np.matmul(np.transpose(A), Y)  # Y^(k+1) = A^(-1) * Y^(k)
        iterations += 1

    return X[0] / previous_X[0], iterations


def method_degree(A, eps):
    previous_Y = previous_eigenvalue = current_eigenvalue = None
    iterations = 0
    Y = np.ones(A.shape[0]) # начальный вектор

    while True:
        if previous_eigenvalue is not None \
                and previous_Y is not None \
                and (current_eigenvalue - previous_eigenvalue) < eps:
            break
        if previous_Y is not None and current_eigenvalue is not None:
            previous_eigenvalue = math.sqrt(
                (np.matmul(Y, Y)) / (np.matmul(previous_Y, previous_Y))
            )  # (Y^(k+1), Y^(k+1)) / (Y^(k), Y^(k))
        previous_Y = Y
        Y = np.matmul(A, Y)  # Y^(k+1) = A * Y^(k)
        current_eigenvalue = math.sqrt(
            (np.matmul(Y, Y)) / (np.matmul(previous_Y, previous_Y))
        )
        iterations += 1

    return Y[0] / previous_Y[0], iterations
