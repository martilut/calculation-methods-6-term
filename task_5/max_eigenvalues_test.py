import numpy as np
import prettytable as pt

from task_1.cond_numbers_test import generate_hilbert_matrix
from task_5.max_eigenvalues import method_scalar, method_degree


def get_table(A):
    true_eigenvalues, _ = np.linalg.eig(A)
    true_max_eigenvalue = max(true_eigenvalues, key=abs)
    print('True max eigenvalue:', true_max_eigenvalue)
    print()
    table = pt.PrettyTable()
    table.field_names = ['Eps', 'Degree eigenvalue', 'Degree error', 'Degree iterations',
                         'Scalar eigenvalue', 'Scalar error', 'Scalar iterations']

    for i in range(-2, -6, -1):
        eps = 10 ** i

        degree_eigenvalue, degree_iterations = method_degree(A, eps)
        scalar_eigenvalue, scalar_iterations = method_scalar(A, eps)

        table.add_row([eps, degree_eigenvalue, abs(true_max_eigenvalue - degree_eigenvalue), degree_iterations,
                       scalar_eigenvalue, abs(true_max_eigenvalue - scalar_eigenvalue), scalar_iterations])

    print(table)


def matrices():
    print("-----------------------------------------------")
    print("Test 1")
    print("-----------------------------------------------")

    A = np.array([[-0.81417, -0.01937, 0.41372],
                  [-0.01937, 0.54414,  0.00590],
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

    A = np.array([[-12, 0, 0, 0],
                  [0, -9, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1]], float)

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
