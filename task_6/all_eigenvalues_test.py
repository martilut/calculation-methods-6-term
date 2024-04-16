import random

import numpy as np
import prettytable as pt

from all_eigenvalues import get_all_eigenvalues
from task_1.cond_numbers_test import generate_hilbert_matrix


def check_eigenvalues(matrix, eigenvalues):
    checks = [False for _ in range(len(eigenvalues))]
    for j in range(len(eigenvalues)):
        eigenvalue = eigenvalues[j]
        for i in range(len(matrix)):
            r = sum([abs(matrix[i][j]) for j in range(len(matrix)) if i != j])
            if abs(eigenvalue - matrix[i][i]) <= r:
                checks[j] = True
                break
    return False not in checks


def get_table(matrix):
    true_eigenvalues = np.linalg.eigvals(matrix)
    true_eigenvalues.sort()

    print('True eigenvalues:', list(true_eigenvalues))

    table = pt.PrettyTable()
    table.field_names = ['Eps', 'Eigenvalues', 'Error', 'Iterations', 'Geršgorin theorem']

    for degree in range(-5, -1):
        eps = 10 ** degree
        eigenvalues, iterations = get_all_eigenvalues(matrix, eps)
        eigenvalues = np.sort(eigenvalues)

        error = np.linalg.norm(eigenvalues - true_eigenvalues)

        check = check_eigenvalues(matrix, eigenvalues)

        table.add_row([eps, list(eigenvalues), error, iterations, check])

    print(table)


def matrices():
    print("-----------------------------------------------")
    print("Test 1")
    print("-----------------------------------------------")

    A = np.array([[-0.81417, -0.01937, 0.41372],
                  [-0.01937, 0.54414, 0.00590],
                  [0.41372, 0.00590, -0.81445]], float)

    print("Matrix:")
    print(A)
    print()

    get_table(A)

    print("-----------------------------------------------")
    print("Test 2: Matrices")
    print("-----------------------------------------------")

    A = np.array([[-0.95121, -0.09779, 0.35843],
                  [-0.09779, 0.61545, 0.02229],
                  [0.35843, 0.02229, -0.95729],
                  ], float)

    print("Matrix:")
    print(A)
    print()

    get_table(A)

    print("-----------------------------------------------")
    print("Test 3: Matrices")
    print("-----------------------------------------------")

    A = np.array([
        [4, -30, 60, -35],
        [-30, 300, -675, 420],
        [60, -675, 1620, -1050],
        [-35, 420, -1050, 700]], dtype=float)

    print("Matrix:")
    print(A)
    print()

    get_table(A)

    print("-----------------------------------------------")
    print("Test 4: Hilbert matrix")
    print("-----------------------------------------------")

    A = generate_hilbert_matrix(4)

    print("Matrix size: 4x4")

    get_table(A)

    print("-----------------------------------------------")
    print("Test 5: Hilbert matrix")
    print("-----------------------------------------------")

    A = generate_hilbert_matrix(10)

    print("Matrix size: 10x10")

    get_table(A)


if __name__ == '__main__':
    matrices()
