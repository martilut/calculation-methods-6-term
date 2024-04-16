import math
import numpy as np


def get_all_eigenvalues(matrix, eps, strategy="abs", max_iterations=1000):
    iterations = 0
    index = 0
    answer_matrix = np.copy(matrix)

    max_value = 1e10

    while max_value > eps and iterations < max_iterations:
        if strategy == "abs":
            max_value, max_row, max_column = select_by_abs(answer_matrix)
        else:
            max_value, max_row, max_column, new_index = select_by_order(answer_matrix, index)
            index = new_index

        answer_matrix = rotate(answer_matrix, max_row, max_column)
        iterations += 1

    return answer_matrix.diagonal(), iterations


def select_by_abs(matrix):
    current_max = matrix[0][1]
    max_row = 0
    max_column = 1

    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            if i != j:
                if abs(matrix[i, j]) > current_max:
                    current_max = abs(matrix[i, j])
                    max_row = i
                    max_column = j

    return current_max, max_row, max_column


def select_by_order(matrix, index):
    new_index = index
    current_index = 0

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            if i != j:
                if current_index == new_index:
                    new_index += 1
                    return abs(matrix[i, j]), i, j, new_index

                current_index += 1

    return abs(matrix[0, 1]), 0, 1, new_index


def rotate(matrix, max_row, max_column):
    new_matrix = np.copy(matrix)
    row_row = new_matrix[max_row][max_row]
    max_value = new_matrix[max_row][max_column]
    column_column = new_matrix[max_column][max_column]

    angle = 0.5 * math.atan2(-2 * max_value, column_column - row_row)
    sin_phi = math.sin(angle)
    cos_phi = math.cos(angle)
    six_2phi = math.sin(2 * angle)
    cos_2phi = math.cos(2 * angle)

    new_matrix[max_row][max_row] = (
            (row_row * cos_phi * cos_phi) + (column_column * sin_phi * sin_phi) + (2 * max_value * cos_phi * sin_phi)
    )

    new_matrix[max_row][max_column] = 0.5 * (column_column - row_row) * six_2phi + max_value * cos_2phi

    new_matrix[max_column][max_column] = (
            (row_row * sin_phi * sin_phi) + (column_column * cos_phi * cos_phi) - (2 * max_value * cos_phi * sin_phi)
    )

    new_matrix[max_column][max_row] = new_matrix[max_row][max_column]

    for i in range(0, new_matrix.shape[0]):
        if i != max_row and i != max_column:
            max_i = new_matrix[i][max_row]
            new_matrix[i][max_row] = new_matrix[i][max_column] * sin_phi + max_i * cos_phi
            new_matrix[i][max_column] = new_matrix[i][max_column] * cos_phi - max_i * sin_phi

    for j in range(0, new_matrix.shape[0]):
        if j != max_row and j != max_column:
            max_j = new_matrix[max_row][j]
            new_matrix[max_row][j] = new_matrix[max_column][j] * sin_phi + max_j * cos_phi
            new_matrix[max_column][j] = new_matrix[max_column][j] * cos_phi - max_j * sin_phi

    return new_matrix
