import numpy
import math
import pylab


def fib(num):
    if num == 0:
        return 0
    if num == 1 or num == 2:
        return 1
    return fib(num - 2) + fib(num - 1)


def find_fib(num):
    n = 0
    fib_num = fib(n)
    while num >= fib_num:
        n += 1
        fib_num = fib(n)
    return n


func_basic = lambda x: x ** 3 - x
func_bis = lambda x: math.log(x) / math.log(2)
func_gold = lambda x: math.log(1 / x) / math.log(1.618)
func_fib = lambda x: find_fib(1 / x)

X = numpy.arange(0, 1, 0.001)
pylab.plot([x for x in X], [func_basic(x) for x in X])
pylab.grid(True)
pylab.show()
