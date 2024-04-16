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
            previous_eigenvalue = (np.matmul(X, Y)) / (np.matmul(previous_X, Y))
        previous_X = X
        X = np.matmul(A, X)
        current_eigenvalue = (np.matmul(X, Y)) / (np.matmul(previous_X, Y))
        Y = np.matmul(np.transpose(A), Y)
        iterations += 1

    return X[0] / previous_X[0], iterations


def method_degree(A, eps):
    previous_X = previous_eigenvalue = current_eigenvalue = None
    iterations = 0
    X = np.ones(A.shape[0])

    while True:
        if previous_eigenvalue is not None \
                and previous_X is not None \
                and (current_eigenvalue - previous_eigenvalue) < eps:
            break
        if previous_X is not None and current_eigenvalue is not None:
            previous_eigenvalue = math.sqrt(
                (np.matmul(X, X)) / (np.matmul(previous_X, previous_X))
            )
        previous_X = X
        X = np.matmul(A, X)
        current_eigenvalue = math.sqrt(
                (np.matmul(X, X)) / (np.matmul(previous_X, previous_X))
            )
        iterations += 1

    return X[0] / previous_X[0], iterations
