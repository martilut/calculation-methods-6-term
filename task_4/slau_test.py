import numpy as np
from slau import simple_iteration_method, seidel_method
import prettytable as pt


def compareMethods(A, b, eps):
    correct_x = np.linalg.solve(A, b)

    x_simple, iterations_simple = simple_iteration_method(A, b, eps)
    x_seidel, iterations_seidel = seidel_method(A, b, eps)

    error_simple = np.linalg.norm(correct_x - x_simple)
    error_seidel = np.linalg.norm(correct_x - x_seidel)

    if len(correct_x) >= 10:
        correct_x = np.array([])
        x_simple = np.array([])
        x_seidel = np.array([])

    return [correct_x, x_simple, iterations_simple, error_simple, x_seidel, iterations_seidel, error_seidel]


def change_eps(A, b):
    table = pt.PrettyTable()
    table.field_names = ['Eps', 'Correct x', 'SI result', 'SI iterations', 'SI error',
                         'Seidel result', 'Seidel iterations', 'Seidel error']

    for i in range(-10, -1):
        eps = 10 ** i
        results = compareMethods(A, b, eps)
        table.add_row([eps] + results)

    print(table)


def generate_matrix(n):
    matrix = np.random.randint(-10, 10, size=(n, n))
    non_diag_sum = 0
    matrix = (matrix + matrix.T) / 2

    for i in range(n):
        for j in range(n):
            if i != j:
                non_diag_sum += abs(matrix[i, j])

    for i in range(n):
        matrix[i, i] = (abs(matrix[i, i]) + non_diag_sum + 1) * (-1 ** i)

    return matrix


def diagonally_dominant_matrices():
    print("-----------------------------------------------")
    print("Test 1: Diagonally dominant matrices")
    print("-----------------------------------------------")

    sizes = [3, 5, 10, 200, 300]

    for size in sizes:
        A = generate_matrix(size)
        b = np.random.randint(-10, 10, size=(size, 1))
        b = np.float64(b)

        if size < 10:
            print(f"Matrix A: {A}")
            print(f"Vector b: {b}")
        else:
            print(f"Matrix A: {size}x{size}")
            print(f"Vector b: {size}x1")

        change_eps(A, b)


def positive_definite_matrix():
    print("-----------------------------------------------")
    print("Test 2: Positive definite matrix")
    print("-----------------------------------------------")

    A = np.array([[2, -1, 2],
                  [-1, 1, -3],
                  [2, -3, 11]], float)

    b = np.array([1, 1, 1], float)

    print(f"Matrix A: {A}")
    print(f"Vector b: {b}")

    change_eps(A, b)


if __name__ == '__main__':
    diagonally_dominant_matrices()
    positive_definite_matrix()
