import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.constants import golden

SIGMA = 0.0001
LIMIT = 5000


def f(x):
    """
    Represents the f(x...) - multivariable function
    :param x: x parameter
    :return: returns the function value
    """
    return (1.5 - x[0] * (1 - x[1])) ** 2 + (2.25 - x[0] * (1 - x[1] ** 2)) ** 2 + (2.625 - x[0] * (1 - x[1] ** 3)) ** 2


def numerical_gradient(x1, x2, dx=1e-6):
    """
    Numerical version of gradient
    :param x1: first parameter
    :param x2: second parameter
    :param dx: dx - function increment
    :return: numerical form of the gradient for given function
    """
    derivative_x1 = (f([x1 + dx, x2]) - f([x1 - dx, x2])) / (2 * dx)
    derivative_x2 = (f([x1, x2 + dx]) - f([x1, x2 - dx])) / (2 * dx)

    return np.array([derivative_x1, derivative_x2])


def analytical_gradient(x1, x2):
    """
    Analytical version of gradient
    :param x1: first parameter
    :param x2: second parameter
    :return: analytical form of the gradient for given function
    """
    derivative_x1 = 2 * (1.5 - x1 * (1 - x2)) * (x2 - 1) + 2 * (2.25 - x1 * (1 - x2 ** 2)) * (x2 ** 2 - 1) \
                    + 2 * (2.625 - x1 * (1 - x2 ** 3)) * (x2 ** 3 - 1)
    derivative_x2 = 2 * (1.5 - x1 * (1 - x2)) * x1 + 2 * (2.25 - x1 * (1 - x2 ** 2)) * 2 * x1 * x2 \
                    + 2 * (2.625 - x1 * (1 - x2 ** 3)) * 3 * (x2 ** 2) * x1

    return np.array([derivative_x1, derivative_x2])


def golden_ratio(f, a, b, e, grad, x):
    """
    One-dimensional search method for step size
    :golden - the Golden ratio const
    :param f: multivariable function f
    :param a: left border
    :param b: right border
    :param e: accuracy
    :param grad: gradient
    :param x: current point
    :return: step size
    """
    t = golden - 1
    x1_n = a + (1 - t) * (b - a)
    x2_n = a + t * (b - a)
    f1_n = f(x - x1_n * grad)
    f2_n = f(x - x2_n * grad)
    e_n = (b - a) / 2

    while True:
        if e_n <= e:
            return (b + a) / 2

        if f1_n <= f2_n:
            b = x2_n
            x2_n = x1_n
            f2_n = f1_n
            x1_n = a + (1 - t) * (b - a)
            f1_n = f(x - x1_n * grad)
        else:
            a = x1_n
            x1_n = x2_n
            f1_n = f2_n
            x2_n = a + t * (b - a)
            f2_n = f(x - x2_n * grad)

        e_n = t * e_n


def vector_search(f, x0, grad, x_, e=SIGMA):
    """
    Search for a minimum of a function in a given direction
    :param f: function
    :param x0: segment begin
    :param grad: gradient
    :param x_: x parameter
    :param e: approximation
    :return: line segment which contains the minimum of the function
    """
    if f(x_ - x0 * grad) > f(x_ - (x0 + e) * grad):
        x = x0 + e
        h = e
    else:
        x = x0 - e
        h = -e
    while True:
        h *= 2
        x_next = x + h

        if f(x_ - x * grad) > f(x_ - x_next * grad):
            x = x_next
        else:
            return [x - h / 2, x_next]


def multivariable_gradient_decent(f, gradient, x, e=SIGMA, lim=LIMIT):
    """
    Multivariable gradient decent is a first-order iterative optimization algorithm
    for finding the minimum of a function. To find a local minimum of a function
    using gradient descent, one takes steps proportional to the negative of the gradient
    (or approximate gradient in case of numerical gradient) of the function at the current point.
    :param f: given function
    :param gradient: gradient method: analytical or numerical
    :param x: current point in format [x1, x2]
    :param e: accuracy
    :param lim: iterations limit
    :return: approximate values of function minimum ([x1, x2]) and the amount of algorithm iterations
    """
    nit = 0
    x_min = x.copy()

    for i in range(0, 5):
        j = 0
        x = x + np.array([30 * i * (-1) ** i] * len(x))
        grad = gradient(x[0], x[1])
        grad = grad / norm(grad)
        borders = vector_search(f, 0, grad, x)
        h = golden_ratio(f=f, a=borders[0], b=borders[1], e=0.00000001, grad=grad, x=x)

        while norm(gradient(x[0], x[1])) > e and j < lim:
            j += 1
            borders = vector_search(f=f, x0=0, grad=grad, x_=x)
            h = golden_ratio(f=f, a=borders[0], b=borders[1], e=0.00000001, grad=grad, x=x)
            x = x - h * grad
            grad = gradient(x[0], x[1])
            grad = grad / norm(grad)
            nit += 1

        if f(x_min) > f(x):
            x_min = x.copy()

    return x_min, nit


tf = open('../output/grad.txt', 'w')
tf.write(str(multivariable_gradient_decent(f=f, gradient=analytical_gradient, x=[100, 100], e=0.001)) + '\n')
tf.write(str(multivariable_gradient_decent(f=f, gradient=numerical_gradient, x=[100, 100], e=0.001)))
tf.close()

count = np.array([0])


def wrap_f(x):
    count[0] += 1
    return f(x)


def wrap_analytical_gradient(x1, x2):
    count[0] += 2
    return analytical_gradient(x1, x2)


def wrap_numerical_gradient(x1, x2):
    count[0] += 2
    return numerical_gradient(x1, x2)


iters = []
counts = []
accuracy = []
x1 = []
x2 = []

for e in np.arange(0.001, 0.02, 0.001):
    count[0] = 0
    res = multivariable_gradient_decent(wrap_f, wrap_analytical_gradient, [0, 0], e=e)
    counts.append(count[0])
    iters.append(res[1])
    accuracy.append(e)
    x1.append(res[0][0])
    x2.append(res[0][1])

data = pd.DataFrame({
    'Колличество итераций': iters,
    'Колличество вычислений': counts,
    'Точность': accuracy,
    'Значение X1': x1,
    'Значение X2': x2,
})

print(data.to_string())
