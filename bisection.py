import math
import pylab
import numpy

func_glob =  lambda x: x ** 3 - x

a1, b1 = 0, 1.0

e = 0.0001


def bisection(a, b, f):
    x = (a + b) / 2
    # math.fabs - returns an absolute value of x
    # could be replaced with
    # fx = 0
    # fx = fx if f(x) > 0 else -1 * fx
    while math.fabs(f(x)) >= e:
        x = (a + b) / 2
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
    return (a + b) / 2


X = numpy.arange(0, 1.0, 0.01)
pylab.plot([x for x in X], [func_glob(x) for x in X])
pylab.grid(True)
pylab.show()

print ('Приближенное значение уравнения: %s' % bisection(a1, b1, func_glob))
