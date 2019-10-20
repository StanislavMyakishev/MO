import math


def line_search(x0, b, func):
    if func(x0) > func(x0 + b):
        x1 = x0 + b
        h = b
    elif func(x0) > func(x0 - b):
        x1 = x0 - b
        h = -b
    h *= 2
    xk = x1
    xk1 = xk + h
    while func(xk) > func(xk1):
        h *= 2
        xk1 = xk1 + h
    return xk, xk1


def func(x):
    res = math.pow(x, 3) - x
    return res


print(line_search(0.8, 0.001, func))
