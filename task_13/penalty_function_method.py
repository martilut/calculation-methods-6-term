import numpy as np
from scipy.optimize import fmin

from task_13.test.functions_1 import n, m_w, f, f_grad, w_list, w_grad_list


class PenaltyFunctionOptimization:
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
        self.alpha = 0.5
        self.beta = 5
        self.eps = 1e-2
        self.max_iter = 1000
        self.iterations_counter = 0
        self.func_calls_counter = 0

    def penalty_function(self, x, p=2):
        result = 0
        for i in range(self.m_w):
            result += abs(self.w_list[i](x)) ** p
        for i in range(self.m_phi):
            result += max(0, self.phi_list[i](x)) ** p
        return result

    def helping_function(self, x):
        return self.f(x) + self.alpha * self.penalty_function(x)

    def penalty_function_method(self):
        self.iterations_counter = 0
        self.func_calls_counter = 0
        self.alpha = 0.5
        x = self.x_0

        while self.iterations_counter < self.max_iter:
            self.iterations_counter += 1
            x_new = fmin(self.helping_function, x, disp=False)
            self.func_calls_counter += 1
            if self.alpha * self.penalty_function(x_new) < self.eps:
                return x_new
            self.alpha *= self.beta
            x = x_new
        return x


def get_penalty_optimizer():
    x_0 = np.array([0.5, 0.2])
    optimizer = PenaltyFunctionOptimization(
        n, m_w, 0, f, f_grad, w_list, w_grad_list, [], [], x_0
    )
    result = optimizer.penalty_function_method()
    return result, f(result), optimizer

