import numpy as np

# Определение функции f(x,y)
def f(x):
    return x[0]**2 + 8*x[1]**2 - x[0]*x[1] + x[0]

# Определение градиента функции f(x,y)
def grad_f(x):
    return np.array([2*x[0]-x[1]+1, 16*x[1]-x[0]])

# Метод Флетчера-Ривса
def fletcher_reeves(f, grad_f, x0, eps1, eps2, max_iter):
    x = x0
    d = -grad_f(x)
    iter_count = 0
    while iter_count < max_iter:
        alpha = golden_section_search(f, x, d, eps2)
        x_prev = x
        x = x + alpha * d
        grad_prev = grad_f(x_prev)
        grad = grad_f(x)
        beta = np.dot(grad, grad) / np.dot(grad_prev, grad_prev)
        d = -grad + beta * d
        grad_norm = np.linalg.norm(grad)
        if grad_norm < eps1:
            break
        iter_count += 1
    return x, iter_count, grad_norm

# Метод золотого сечения для поиска оптимального шага
def golden_section_search(f, x, d, eps):
    phi = (1 + np.sqrt(5)) / 2
    a = 0
    b = 1
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    while abs(b - a) > eps:
        if f(x + x1 * d) < f(x + x2 * d):
            b = x2
        else:
            a = x1
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi
    return (a + b) / 2

# Тестирование метода Флетчера-Ривса на функции f(x,y)
x0 = np.array([1.5, 0.1])
eps1 = 1e-6
eps2 = 1e-6
max_iter = 1000

x_min, iter_count, grad_norm = fletcher_reeves(f, grad_f, x0, eps1, eps2, max_iter)

print("Минимум функции:", x_min)
print("Число итераций:", iter_count)
print("Норма градиента:", grad_norm)
