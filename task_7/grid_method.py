import numpy as np


def get_grid(A, h, cells):
    grid = []
    for i in range(0, cells + 1):
        node = A - 0.5 * h + i * h
        grid.append(node)
    return grid


def refine_grid(grid):
    if len(grid) == 0 or len(grid) == 1:
        return list(grid)

    new_grid = []

    for i in range(0, len(grid) - 1):
        h = 0.5 * (grid[i + 1] - grid[i])
        new_grid.append(grid[i])
        new_grid.append(grid[i] + h)

    new_grid.append(grid[-1])
    return new_grid


def get_diagonals(qx, rx, grid, h, cells, a1, a2, b1, b2):
    lowerDiag = []
    mainDiag = [a1 / 2.0 + a2 / float(h)]
    upperDiag = [a1 / 2.0 - a2 / float(h)]

    for i in range(1, cells):
        qi = qx(grid[i])
        ri = rx(grid[i])
        lowerDiag.append(1 / float(h ** 2) - qi / float(2 * h))
        mainDiag.append(-2 / float(h ** 2) - ri)
        upperDiag.append(1 / float(h ** 2) + qi / float(2 * h))

    lowerDiag.append(b1 / 2.0 - b2 / float(h))
    mainDiag.append(b1 / 2.0 + b2 / float(h))

    return lowerDiag, mainDiag, upperDiag


def get_result_vector(fx, grid, cells, a, b):
    result_vector = [a]
    for i in range(1, cells):
        result_vector.append(fx(grid[i]))
    result_vector.append(b)
    return result_vector


def solve(qx, rx, fx, A, B, a1, a2, a, b1, b2, b, cells):
    h = (B - A) / float(cells)
    grid = get_grid(A, h, cells)
    #print(grid)
    lower_diagonal, main_diagonal, upper_diagonal \
        = get_diagonals(qx, rx, grid, h, cells, a1, a2, b1, b2)
    result_vector = get_result_vector(fx, grid, cells, a, b)
    result = list(get_tridiagonal_matrix(lower_diagonal, main_diagonal, upper_diagonal, result_vector))
    #print(result)
    return result, grid


def refine_result(v1, v2):
    result = []
    for i in range(0, len(v2)):
        result.append(v2[i] + calc_delta(v1, v2, i))
    return result


def calc_delta(v1, v2, num):
    if num % 2 != 0:
        prev_delta = calc_delta(v1, v2, num - 1)
        next_delta = calc_delta(v1, v2, num + 1)
        return 0.5 * (prev_delta + next_delta)

    return (v2[num] - v1[int(num / 2)]) / 3.0


def get_norm(v1, v2):
    deltas = []
    for i in range(0, len(v2)):
        deltas.append(abs(calc_delta(v1, v2, i)))
    return max(deltas)


def solve_eps(qx, rx, fx, A, B, a1, a2, a, b1, b2, b, eps):
    cells = 4
    h = (B - A) / float(cells)
    v1, grid = solve(qx, rx, fx, A, B, a1, a2, a, b1, b2, b, cells)
    deltas = []
    norms = []
    while True:
        h = h * 0.5
        cells = cells * 2
        v2, grid = solve(qx, rx, fx, A, B, a1, a2, a, b1, b2, b, cells)
        norm = get_norm(v1, v2)
        deltas.append(h)
        norms.append(norm)
        #print(norm)
        if norm <= eps:
            break
        v1 = v2
    return refine_result(v1, v2), deltas, norms, grid


def get_tridiagonal_matrix(a, b, c, d):
    n = len(d)
    w = np.zeros(n - 1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])

    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])

    p[n - 1] = g[n - 1]

    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p
