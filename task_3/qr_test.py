import numpy as np

from task_1.cond_numbers_test import generate_hilbert_matrix
from task_1.cond_numbers import calculate_cond_s
from task_3.qr_calculation import calculate_QR


def well_conditioned_matrices():
    print("---------------------------------------------------------------")
    print("Test 1: Well-conditioned matrices")
    print("---------------------------------------------------------------")

    A = np.array([[1, 2],
                  [2, 1]])
    b = np.array([1, 1])
    print_results(A, b)

    A = np.array([[400, 350],
                  [1200, 600]])
    b = np.array([1, 1])
    print_results(A, b)

    A = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], float)
    b = np.array([1, 1, 1, 1], float)
    print_results(A, b)


def poorly_conditioned_matrices():
    print("---------------------------------------------------------------")
    print("Test 2: Poorly-conditioned matrices")
    print("---------------------------------------------------------------")

    A = np.array([[1.00, 0.99],
                  [0.99, 0.98]])
    b = np.array([2, 2])
    print_results(A, b)

    A = np.array([
        [-400.60, 199.80],
        [1198.80, -600.40]
    ])
    b = np.array([200, -600])
    print_results(A, b)

    A = np.array([
        [-401.98, 200.34],
        [1202.04, -602.32]
    ])
    b = np.array([199, -601])

    print_results(A, b)


def hilbert_matrices():
    print("---------------------------------------------------------------")
    print("Test 3: Hilbert matrices")
    print("---------------------------------------------------------------")

    A = generate_hilbert_matrix(3)
    b = np.array([1. for _ in range(3)], dtype=float)
    print_results(A, b)

    A = generate_hilbert_matrix(4)
    b = np.array([1. for _ in range(4)], dtype=float)
    print_results(A, b)

    A = generate_hilbert_matrix(7)
    b = np.array([1. for _ in range(7)], dtype=float)
    print_results(A, b)


def print_results(A, b):
    print(f"R: {A}")
    print(f"b: {b}")

    Q, R, X = calculate_QR(A, b)
    correct_x = np.linalg.solve(A, b)
    error = np.linalg.norm(X - correct_x)

    cond_A = calculate_cond_s(A)
    cond_Q = calculate_cond_s(Q)
    cond_R = calculate_cond_s(R)

    print()
    print(f"QR x: {X}")
    print(f"Correct x: {correct_x}")
    print('error:', error)
    print('cond_s R =', cond_A)
    print('cond_s Q =', cond_Q)
    print('cond_s R =', cond_R)
    print("---------------------------------------------------------------")
    print()


if __name__ == '__main__':
    well_conditioned_matrices()
    poorly_conditioned_matrices()
    hilbert_matrices()
