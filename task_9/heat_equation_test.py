from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np

from task_9.heat_equation import ex_scheme, im_scheme


def draw_graphs(u, f, x_steps, time_steps, x_limit, time_limit, k):
    _, subplots = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(15, 10))
    x_grid_nodes1, time_grid_nodes1, u_values1 = \
        ex_scheme(x_steps, time_steps, u, k, f, x_limit, time_limit)
    x_grid_nodes2, time_grid_nodes2, u_values2 = \
        im_scheme(x_steps, time_steps, u, k, f, x_limit, time_limit)
    x, time = np.meshgrid(x_grid_nodes1, time_grid_nodes1)
    add_data(subplots[0], x, time, u_values1, 'Явная схема')
    add_data(subplots[1], x, time, u_values2, 'Неявная схема')
    plt.show()


def add_data(plot, x, t, z, title):
    plot.plot_surface(x, t, z)
    plot.set_title(title)
    plot.set_xlabel('x')
    plot.set_ylabel('t')
    plot.set_zlabel('z')


def function1():
    k = 0.001
    u = lambda x, t: (x ** 3) + (t ** 3)
    f = lambda x, t: t * (x ** 2) - k * (t ** 3) * x
    steps = 20
    x_limit = 1
    time_limit = 1

    draw_graphs(u, f, steps, steps, x_limit, time_limit, k)


def function2():
    k = 0.01
    u = lambda x, t: (x ** 3) * (t ** 3)
    f = lambda x, t: x * (t ** 2) - k * (x ** 3) * t
    steps = 30
    x_limit = 1
    time_limit = 3

    draw_graphs(u, f, steps, steps, x_limit, time_limit, k)


def function5():
    k = 0.1
    u = lambda x, t: (x ** 3) * (t ** 3)
    f = lambda x, t: x * (t ** 2) - k * (x ** 3) * t
    steps = 30
    x_limit = 1
    time_limit = 5

    draw_graphs(u, f, steps, steps, x_limit, time_limit, k)


def function3():
    k = 0.01
    u = lambda x, t: sin(2 * t + 1) * cos(2 * x)
    f = lambda x, t: sin(t) * cos(x) ** 3 - 2 * sin(x) ** 2 * cos(x)
    steps = 100
    x_limit = 3
    time_limit = 3

    draw_graphs(u, f, steps, steps, x_limit, time_limit, k)


def function4():
    k = 0.01
    u = lambda x, t: sin(2 * t + 1) + cos(2 * x)
    f = lambda x, t: cos(x) * sin(t)
    steps = 100
    x_limit = 3
    time_limit = 5

    draw_graphs(u, f, steps, steps, x_limit, time_limit, k)


def function6():
    k = 0.1
    u = lambda x, t: sin(2 * t + 1) + cos(2 * x)
    f = lambda x, t: cos(x) * sin(t)
    steps = 100
    x_limit = 3
    time_limit = 5

    draw_graphs(u, f, steps, steps, x_limit, time_limit, k)


if __name__ == '__main__':
    function1()
    function2()
    function5()
    function3()
    function4()
    function6()
