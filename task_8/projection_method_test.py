import math
from math import exp, sqrt

import matplotlib.pyplot as plt
import numpy as np

from galerkin_method import galerkin_method
from task_8.collocation_method import collocation_method


def compare(a, b, p, q, r, f, N, true_result):
    nodes = np.linspace(a, b, 100)
    colors = ["blue", "red", "green", "yellow"]
    for n in range(2, N + 1):
        actual = galerkin_method(a, b, p, q, r, f, n)
        approx_sol = collocation_method(lambda x: q(x) / p(x), lambda x: r(x) / p(x), lambda x: f(x)/p(x), a, b, 0.0, 0.0, N=n)
        plt.plot(nodes, actual(nodes), label=f'N={n}', color=colors[n-2])
        plt.plot(nodes, approx_sol(nodes), linestyle='dashed', color=colors[n-2])
    true_result = np.vectorize(true_result)
    plt.plot(nodes, true_result(nodes), label=f'Exact solution', color="black")
    plt.legend()
    plt.show()


def func_1():
    p = lambda x: x - 1
    q = lambda x: -x
    r = lambda x: 1
    f = lambda x: (x - 1) ** 2

    expected = lambda x: -((x - 1) ** 2 - 4 * exp(x + 1) + math.e ** 2 * (x + 1) ** 2) / (1 + math.e ** 2)
    compare(-1, 1, p, q, r, f, 5, expected)


def func_2():
    p = lambda x: 1
    q = lambda x: x
    r = lambda x: 0
    f = lambda x: -2 * x

    expected = lambda x: -2 * x + (2 * math.erf(x / sqrt(2))) / math.erf(1 / sqrt(2))
    compare(-1, 1, p, q, r, f, 5, expected)


if __name__ == '__main__':
    func_1()
    func_2()
