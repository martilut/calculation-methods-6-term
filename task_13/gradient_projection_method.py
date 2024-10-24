import numpy as np
from task_13.test.functions_1 import n, m_w, f, f_grad, w_list, w_grad_list


def get_grad_optimizer():
    optimizer = GradientProjectionOptimization(n, m_w, f, f_grad, w_list, w_grad_list, [0.8, 0.2])
    result = optimizer.gradient_projection_method()
    return result, f(result), optimizer


class GradientProjectionOptimization:
    def __init__(self, n, m, f, f_grad, w_list, w_grad_list, x_0):
        self.n = n
        self.m = m
        self.f = f
        self.f_grad = f_grad
        self.w_list = w_list
        self.w_grad_list = w_grad_list
        self.x_0 = x_0
        self.h_0 = 1
        self.eps = 1e-2
        self.max_iter = 1000
        self.iterations_counter = 0
        self.func_calls_counter = 0

    def find_lamdas(self, x):
        A = np.zeros(shape=(self.m, self.m))
        for k in range(self.m):
            w_k = self.w_grad_list[k]
            for p in range(self.m):
                w_p = self.w_grad_list[p]
                coeff = 0
                for i in range(self.n):
                    coeff += w_k(x)[i] * w_p(x)[i]
                A[k][p] = coeff
        b = np.zeros(self.m)
        for k in range(self.m):
            w_k = self.w_grad_list[k]
            value = 0
            for i in range(self.n):
                value += w_k(x)[i] * self.f_grad(x)[i]
            b[k] = -value
        return np.linalg.solve(A, b)

    def find_gradient_projection(self, x):
        lambdas = self.find_lamdas(x)
        gradient_projection = np.zeros(self.n)
        for i in range(self.n):
            numerator = self.f_grad(x)[i]
            for j in range(self.m):
                numerator += lambdas[j] * self.w_grad_list[j](x)[i]
            denominator = 0
            for k in range(self.n):
                sum_denominator = self.f_grad(x)[i]
                for j in range(self.m):
                    sum_denominator += lambdas[j] * self.w_grad_list[j](x)[i]
                denominator += sum_denominator ** 2
            denominator = np.sqrt(denominator)
            if denominator < self.eps:
                return None
            gradient_projection[i] = numerator / denominator
        return gradient_projection

    def update_h(self, x, h, gradient_projection):
        h_new = h / 2
        x_new_1 = x - h_new * gradient_projection
        x_new_2 = x + h_new * gradient_projection
        f_x_new_1 = self.f(x_new_1)
        f_x_new_2 = self.f(x_new_2)
        self.func_calls_counter += 1
        if np.abs(f_x_new_1) < np.abs(f_x_new_2):
            return x_new_1, f_x_new_1, h_new
        else:
            return x_new_2, f_x_new_2, h_new

    def gradient_projection_method(self):
        self.iterations_counter = 0
        self.func_calls_counter = 0
        x = self.x_0
        h = self.h_0

        while self.iterations_counter < self.max_iter:
            self.iterations_counter += 1
            gradient_projection = self.find_gradient_projection(x)
            if gradient_projection is None:
                return x
            x_new = x - h * gradient_projection
            f_x_new = self.f(x_new)
            f_x = self.f(x)
            self.func_calls_counter += 3
            while np.abs(f_x_new) > np.abs(f_x) and self.iterations_counter < self.max_iter:
                self.iterations_counter += 1
                x_new, f_x_new, h = self.update_h(x, h, gradient_projection)
            else:
                x = x_new
        return x
