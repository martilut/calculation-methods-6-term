import numpy as np


def get_grid(x_steps, time_steps, u, x_limit, time_limit):
    x_grid_nodes = np.linspace(0, x_limit, x_steps + 1)
    time_grid_nodes = np.linspace(0, time_limit, time_steps + 1)

    u_values = np.zeros((x_steps + 1, time_steps + 1))

    for i in range(x_steps + 1):
        u_values[i, 0] = u(x_grid_nodes[i], 0)

    for i in range(time_steps + 1):
        u_values[0, i] = u(x_grid_nodes[0], time_grid_nodes[i])
        u_values[x_steps, i] = u(x_grid_nodes[x_steps], time_grid_nodes[i])

    return x_grid_nodes, time_grid_nodes, u_values


def ex_scheme(x_steps, time_steps, u, k, f, x_limit, time_limit):
    h = x_limit / x_steps
    tau = time_limit / time_steps
    print(2 * k * tau, h ** 2)
    if 2 * k * tau > h ** 2:
        print('Явная схема неустойчива')
    print("------------------------------------------------------")

    x_grid_nodes, time_grid_nodes, u_values = get_grid(x_steps, time_steps, u, x_limit, time_limit)

    for time in range(1, time_steps + 1):
        for x in range(1, x_steps):
            diff = u_values[x - 1, time - 1] - 2 * u_values[x, time - 1] + u_values[x + 1, time - 1]
            u_values[x, time] = u_values[x, time - 1] \
                             + tau * (k / (h ** 2) * diff + f(x_grid_nodes[x], time_grid_nodes[time - 1]))

    return x_grid_nodes, time_grid_nodes, u_values


def im_scheme(x_steps, time_steps, u, k, f, x_limit, time_limit):
    h = x_limit / x_steps
    tau = time_limit / time_steps

    x_grid_nodes, time_grid_nodes, u_values = get_grid(x_steps, time_steps, u, x_limit, time_limit)

    for t in range(1, time_steps + 1):
        m = np.zeros((x_steps + 1, x_steps + 1))
        v = np.zeros(x_steps + 1)

        m[0, 0] = -(tau * k / h + 1)
        m[0, 1] = tau * k / h
        m[x_steps, x_steps - 1] = -tau * k / h
        m[x_steps, x_steps] = tau * k / h - 1

        v[0] = -u_values[0, t - 1] - tau * f(x_grid_nodes[0], time_grid_nodes[t])
        v[x_steps] = -u_values[x_steps, t - 1] - tau * f(x_grid_nodes[x_steps], time_grid_nodes[t])

        for x in range(1, x_steps):
            m[x, x - 1] = m[x, x + 1] = tau * k / (h ** 2)
            m[x, x] = -2 * (tau * k / (h ** 2)) - 1
            v[x] = -u_values[x, t - 1] - tau * f(x_grid_nodes[x], time_grid_nodes[t])

        u_values[:, t] = np.linalg.solve(m, v)

    return x_grid_nodes, time_grid_nodes, u_values
