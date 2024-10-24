def f(x):
    return x[0] ** 2 + x[1] ** 2


def f_grad(x):
    return [2 * x[0], 2 * x[1]]


def w_1(x):
    return x[0] + x[1] - 1


def grad_w_1(x):
    return [1, 1]


def phi_1(x):
    return x[0] - x[1]


def grad_phi_1(x):
    return [1, -1]


n = 2
m_w = 1
m_phi = 1

w_list = [w_1]
w_grad_list = [grad_w_1]  # w_grad_list[0][0](x) = gradient of w_1 with respect to x at point x

phi_list = [phi_1]
phi_grad_list = [grad_phi_1]  # phi_grad_list[0][0](x) = gradient of phi_1 with respect to x at point x
