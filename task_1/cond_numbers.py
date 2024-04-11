import numpy as np
import math


def calculate_cond_s(A):
    inv_A = np.linalg.inv(A)
    norm_A = np.linalg.norm(A)
    norm_inv = np.linalg.norm(inv_A)

    return norm_A * norm_inv


def calculate_cond_v(A):
    numerator = 1
    for n in range(A.shape[0]):
        root = 0
        for m in range(A.shape[0]):
            root += A[n, m] ** 2
        numerator *= math.sqrt(root)
    return numerator / abs(np.linalg.det(A))


def calculate_cond_a(A):
    max_value = 0
    inv_A = np.linalg.inv(A)
    for i in range(A.shape[0]):
        a_n = A[i]
        c_n = inv_A[:, i]
        product = np.linalg.norm(a_n) * np.linalg.norm(c_n)
        if product > max_value:
            max_value = product

    return max_value
