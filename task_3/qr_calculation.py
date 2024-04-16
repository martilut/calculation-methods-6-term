import math

import numpy as np


def calculate_QR(A, b):
    Q, R = rotation_method(A=A)
    answer = np.dot(np.transpose(Q), b)
    X = np.zeros(R.shape[0])

    # обратный ход метода Гаусса, R - верхняя треугольная матрица
    for i in range(R.shape[0] - 1, -1, -1):
        value = answer[i]

        for j in range(i + 1, R.shape[0]):
            value -= R[i, j] * X[j]

        X[i] = value / R[i, i]

    return Q, R, X


def rotation_method(A):
    n = A.shape[0]
    Q = np.eye(n)
    R = A.copy()
    for j in range(n):
        for i in range(n - 1, j, -1):
            T = rotation_matrix(n, i, j, R[i - 1, j], R[i, j])
            R = np.matmul(np.transpose(T), R)
            Q = np.matmul(Q, T)
    return Q, R


def rotation_matrix(size, i, j, t, s):
    if t == 0 and s == 0:
        sin = 0
        cos = 1
    else:
        root = math.sqrt(t ** 2 + s ** 2)
        sin = s / root
        cos = -t / root

    result_matrix = np.eye(size)
    result_matrix[i, i] = result_matrix[j, j] = cos
    result_matrix[i, j] = -sin
    result_matrix[j, i] = sin

    return result_matrix
