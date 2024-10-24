import numpy as np

from task_13.test.functions_1 import f
from task_13.test.functions_3 import m_phi, phi_list


def check_value(x):
    for i in range(m_phi):
        if phi_list[i](x) >= 0:
            return False
    return True

def compute_gradient(u, x0, dx=1e-2):
    n = len(x0)
    gradient = np.zeros(n)
    for i in range(n):
        delta = np.array([0 if j != i else dx for j in range(n)])
        gradient[i] = (u(x0 + delta) - u(x0 - delta)) / (2 * dx)
    return gradient


def gradient_descent(
        f, start, learning_rate=0.0024, max_iters=50, eps=1e-2
):
    num_of_iters = 0
    x0 = start
    gradient = compute_gradient(f, x0)
    x1 = x0 - learning_rate * compute_gradient(f, x0)
    if not check_value(x1):
        return x0, num_of_iters, False
    while np.linalg.norm(x0 - x1) > eps and num_of_iters < max_iters:
        num_of_iters += 1
        x0 = x1
        gradient = compute_gradient(f, x0)
        x1 = x0 - learning_rate * gradient
        if not check_value(x1):
            return x0, num_of_iters, False
    return x0, num_of_iters, True


if __name__ == '__main__':
    start = np.array([3., 4.])
    learn_rate = 0.1
    result = gradient_descent(f, start, learn_rate)
    print(result)
    print(f(result[0]))
