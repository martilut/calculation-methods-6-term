from task_1.cond_numbers import *
import matplotlib.pyplot as plt
import numpy as np


def generate_hilbert_matrix(n, round_to=None):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 1 / (i + j + 1)
            if round_to is not None:
                matrix[i, j] = round(matrix[i, j], round_to)
    return matrix


def variate_matrix(matrix, value=1e-2):
    return matrix + value


def generate_tridiagonal_matrix(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or abs(i - j) == 1:
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    return matrix


def calculate_matrix(A, degrees, B=None, x=None, index=1, path=None):
    if B is None:
        B = A @ x
    if x is None:
        x = np.linalg.solve(A, B)

    cond_s_list = []
    cond_v_list = []
    cond_a_list = []
    error_list = []

    for degree in degrees:
        e = 10 ** degree
        new_A = A - e * np.eye(A.shape[0])
        new_B = B - e

        cond_s = calculate_cond_s(new_A)
        cond_v = calculate_cond_v(new_A)
        cond_a = calculate_cond_a(new_A)

        cond_s_list.append(cond_s)
        cond_v_list.append(cond_v)
        cond_a_list.append(cond_a)

        new_result = np.linalg.solve(new_A, new_B)
        error = np.linalg.norm(x - new_result)
        error_list.append(error)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=80)
    axs[0, 0].plot(degrees, cond_s_list, label='cond_s')
    axs[0, 1].plot(degrees, cond_v_list, label='cond_v')
    axs[1, 0].plot(degrees, cond_a_list, label='cond_a')
    axs[1, 1].plot(degrees, error_list, label='error')
    for i in range(2):
        for j in range(2):
            axs[i, j].set_xlabel('eps=10^(x)')
            axs[i, j].set_ylabel('value')
            axs[i, j].legend()

    if path is not None:
        plt.savefig(f"{path}/matrix_{index}.png")


def analyze_book_matrices(degrees):
    matrices = [
        np.array([
            [-400.60, 199.80],
            [1198.80, -600.40]
        ]),
        np.array([
            [-401.98, 200.34],
            [1202.04, -602.32]
        ]),
    ]
    B = np.array([200, -600])
    for i, matrix in enumerate(matrices):
        calculate_matrix(matrix, degrees, B=B, index=i + 1, path="plots/book_matrices")


def analyze_hilbert_matrices(degrees):
    for n in range(3, 11):
        A = generate_hilbert_matrix(n)
        x = np.ones((n, 1))
        calculate_matrix(A, degrees, x=x, index=n, path="plots/hilbert_matrices")


def analyze_tridiagonal_matrices(degrees):
    for n in range(3, 6):
        A = generate_tridiagonal_matrix(n)
        x = np.ones((n, 1))
        calculate_matrix(A, degrees, x=x, index=n, path="plots/tridiagonal_matrices")


if __name__ == "__main__":
    degrees = [i for i in range(-2, -10, -1)]
    analyze_book_matrices(degrees)
    analyze_hilbert_matrices(degrees)
    analyze_tridiagonal_matrices(degrees)
