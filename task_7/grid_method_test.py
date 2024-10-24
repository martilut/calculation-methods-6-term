import math
import prettytable as pt
import matplotlib.pyplot as plt

from task_7.grid_method import solve_eps


def pretty_print(deltas, norms):
    table = pt.PrettyTable()
    table.field_names = ["h", "precision"]
    for i in range(0, len(deltas)):
        table.add_row([deltas[i], norms[i]])
    print(table)


def show_function(grid, result):
    plt.plot(grid, result)
    plt.show()
    plt.close()


def func_1():
    qx = lambda x: (1 + x / 2) * (3 - x)
    rx = lambda x: (-math.e ** (x / 2)) * (3 - x)
    fx = lambda x: (2 - x) * (3 - x)
    eps = 0.001
    A = -1
    B = 1
    a1 = 1
    a2 = 0
    a = 0
    b1 = 1
    b2 = 0
    b = 0

    result, deltas, norms, grid = solve_eps(qx, rx, fx, A, B, a1, a2, a, b1, b2, b, eps)
    pretty_print(deltas, norms)
    show_function(grid, result)


def func_2():
    qx = lambda x: x * (x + 3)
    rx = lambda x: math.log(2 + x) * (-1) * (x + 3)
    fx = lambda x: (1 - x / 2) * (-1) * (x + 3)
    eps = 0.001
    A = -1
    B = 1
    a1 = 0
    a2 = 1
    a = 0
    b1 = 1 / 2
    b2 = 1
    b = 0

    result, deltas, norms, grid = solve_eps(qx, rx, fx, A, B, a1, a2, a, b1, b2, b, eps)
    pretty_print(deltas, norms)
    show_function(grid, result)


def func_3():
    qx = lambda x: x * (x + 2) / (x - 2)
    rx = lambda x: (1 - math.sin(x)) * (x + 2) / (x - 2)
    fx = lambda x: (x ** 2) * (x + 2) / (x - 2)
    eps = 0.001
    A = -1
    B = 1
    a1 = 1
    a2 = 0
    a = 0
    b1 = 1
    b2 = 0
    b = 0

    result, deltas, norms, grid = solve_eps(qx, rx, fx, A, B, a1, a2, a, b1, b2, b, eps)
    pretty_print(deltas, norms)
    show_function(grid, result)


if __name__ == '__main__':
    func_1()
    func_2()
    func_3()
