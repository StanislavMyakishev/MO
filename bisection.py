import math

a1, b1 = 0, 1.0
e = 0.0001

func_glob = lambda x: x ** 3 - x


def bisection(a, b, f):
    text_file = open('../output/bis.txt', 'w')
    count = 0
    while b - a >= e:
        x = (a + b) / 2
        d = e/2
        count += 1
        x1 = x - d
        x2 = x + d
        s = ('%d iteration, f(x) = %f, x1 = %f, x2 = %f, x = %f, a = %f, b = %f\n' % (count, f(x), x1, x2, x, a, b))
        text_file.write(s)
        a, b = (a, x1) if f(x1) < f(x2) else (x2, b)
        if count == 100:
            break
    text_file.write('Iterations: %d\n' % count)
    text_file.write('Result: %f\n' % ((a + b)/2))
    text_file.close()
    return (a + b) / 2


print('Приближенное значение уравнения: %s' % bisection(a1, b1, func_glob))
