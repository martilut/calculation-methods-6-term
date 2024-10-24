def f(x):
    return (x[0] - 2) ** 4 + (x[0] - 2 * x[1]) ** 2


def f_grad(x):
    return [4 * (x[0] - 2) ** 3 + 2 * (x[0] - 2 * x[1]), -4 * (x[0] - 2 * x[1])]


def phi_1(x):
    return x[0] ** 2 - x[1]


def grad_phi_1(x):
    return [2 * x[0], -1]


n = 2
m_w = 0
m_phi = 1

w_list = []
w_grad_list = []

phi_list = [phi_1]
phi_grad_list = [grad_phi_1]
