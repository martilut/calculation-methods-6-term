def f(x):
    return (1 - x[0]) ** 2 + 5 * (x[1] - x[0] ** 2) ** 2


def f_grad(x):
    return [-2 * (1 - x[0]) - 20 * x[0] * (x[1] - x[0] ** 2), 10 * (x[1] - x[0] ** 2)]


def w_1(x):
    return x[1] - (2 / 3) * x[0] + 4


def grad_w_1(x):
    return [-2 / 3, 1]


n = 2
m_w = 1
m_phi = 0

w_list = [w_1]
w_grad_list = [grad_w_1]

phi_list = []
phi_grad_list = []
