import numpy as np

from task_13.gradient_descent import gradient_descent
from task_13.test.functions_3 import n, m_w, m_phi, f, f_grad, w_list, w_grad_list, phi_list, phi_grad_list


class BarrierFunctionOptimization:
    def __init__(self, n, m_w, m_phi, f, f_grad, w_list, w_grad_list, phi_list, phi_grad_list, x_0):
        self.n = n
        self.m_w = m_w
        self.m_phi = m_phi
        self.f = f
        self.f_grad = f_grad
        self.w_list = w_list
        self.w_grad_list = w_grad_list
        self.phi_list = phi_list
        self.phi_grad_list = phi_grad_list
        self.x_0 = x_0
        self.mu = 10
        self.beta = 0.1
        self.eps = 1e-2
        self.max_iter = 1000
        self.iterations_counter = 0
        self.func_calls_counter = 0

    def barrier_function(self, x):
        result = 0
        for i in range(self.m_phi):
            result += -1 / self.phi_list[i](x)
        return result

    def helping_function(self, x):
        return self.f(x) + self.mu * self.barrier_function(x)

    def check_value(self, x):
        for i in range(self.m_phi):
            if self.phi_list[i](x) >= 0:
                return False
        return True

    def barrier_function_method(self):
        self.iterations_counter = 0
        self.func_calls_counter = 0
        self.mu = 10
        x = self.x_0

        while self.iterations_counter < self.max_iter:
            self.iterations_counter += 1
            x_new, iters, to_continue = gradient_descent(self.helping_function, x)
            self.func_calls_counter += 1
            if not to_continue:
                return x
            if self.mu * self.barrier_function(x_new) < self.eps:
                return x_new
            self.mu *= self.beta
            x = x_new
        return x

def get_barrier_optimizer():
    x_0 = np.array([0., 1.])
    optimizer = BarrierFunctionOptimization(
        n, m_w, m_phi, f, f_grad, w_list, w_grad_list, phi_list, phi_grad_list, x_0
    )
    result = optimizer.barrier_function_method()
    return result, f(result), optimizer
