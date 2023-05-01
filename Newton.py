import numpy as np

# Определение функции f(x,y)
def f(x):
    return x[0]**2 + 8*x[1]**2 - x[0]*x[1] + x[0]

# Определение градиента функции f(x,y)
def grad_f(x):
    return np.array([2*x[0]-x[1]+1, 16*x[1]-x[0]])

# Определение матрицы Гессе функции f(x,y)
def hess_f(x):
    return np.array([[2, -1], [-1, 16]])

def newton_method(f, grad_f, hess_f, x0, eps1, eps2, max_iter):
    x = x0
    iter_count = 0
    while iter_count < max_iter:
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < eps1:
            break
        hess = hess_f(x)
        d = -np.linalg.solve(hess, grad)
        alpha = 1
        x_new = x + alpha * d
        while f(x_new) > f(x):
            alpha /= 2
            x_new = x + alpha * d
        x = x_new
        iter_count += 1
    return x, iter_count, grad_norm

# Тестирование метода Ньютона на функции f(x,y)
x0 = np.array([1.5, 0.1])
eps1 = 1e-6
eps2 = 1e-6
max_iter = 1000

x_min, iter_count, grad_norm = newton_method(f, grad_f, hess_f, x0, eps1, eps2, max_iter)

print("Минимум функции:", x_min)
print("Число итераций:", iter_count)
print("Норма градиента:", grad_norm)
