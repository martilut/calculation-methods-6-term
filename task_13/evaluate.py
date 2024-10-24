from gradient_projection_method import get_grad_optimizer
from penalty_function_method import get_penalty_optimizer
from barrier_method import get_barrier_optimizer


def print_info(result, f_result, optimizer, true_x, true_f_x):
    print(f"Evaluation x: {result}")
    print(f"Evaluation f(x): {f_result}")
    print(f"True x: {true_x}")
    print(f"True F(x): {true_f_x}")
    print(f"|f(x) - F(x)|: {abs(f_result - true_f_x)}")
    print(f"Iterations count: {optimizer.iterations_counter}")
    print(f"Function calls count: {optimizer.func_calls_counter}")


if __name__ == "__main__":
    print("---Gradient projection method---")
    print("f(x) = x1 ** 2 + x2 ** 2")
    print("w(x) = x1 + x2 - 1 = 0")
    result, f_result, optimizer = get_grad_optimizer()
    true_x, true_f_x = [0.5, 0.5], 0.5
    print_info(result, f_result, optimizer, true_x, true_f_x)
    print()
    print("---Penalty function method---")
    print("f(x) = x1 ** 2 + x2 ** 2")
    print("w(x) = x1 + x2 - 1 = 0")
    result, f_result, optimizer = get_penalty_optimizer()
    true_x, true_f_x = [0.5, 0.5], 0.5
    print_info(result, f_result, optimizer, true_x, true_f_x)
    print()
    print("---Barrier function method---")
    print("f(x) = (x1 - 2) ** 4 + (x1 - 2 * x2) ** 2")
    print("phi(x) = x1 ** 2 - x2 <= 0")
    result, f_result, optimizer = get_barrier_optimizer()
    true_x, true_f_x = [0.96, 0.94], optimizer.f([0.96, 0.94])
    print_info(result, f_result, optimizer, true_x, true_f_x)
