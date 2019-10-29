import math

def fucntion(x):
    res = math.pow(x, 3) - x
    return res


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


def fibonacci(e, a, b):
    text_file = open('../output/fib.txt', 'w')
    F = list()  # список наши Fn , Fn+1 , Fn+2
    text_file.write('((b-a)/e)=')
    r = (b - a) / e
    text_file.write(str("%i \n" % r))
    n = 1
    znach = math.pow(((1 + math.sqrt(5)) / 2), n) / math.sqrt(5)
    while znach < r:
        n += 1
        znach = math.pow(((1 + math.sqrt(5)) / 2), n) / math.sqrt(5)
    text_file.write('number of iterations n+2:')
    text_file.write(str("%i \n" % n))
    text_file.write('F(n)=')
    F.append(math.pow(((1 + math.sqrt(5)) / 2), n - 2) / math.sqrt(5))  # F(n)
    text_file.write(str("%.4f \n" % F[0]))
    text_file.write('F(n+1)=')
    F.append(math.pow(((1 + math.sqrt(5)) / 2), n - 1) / math.sqrt(5))  # F(n+1)
    text_file.write(str("%.4f \n" % F[1]))
    text_file.write('F(n+2)=')
    F.append(znach)  # F(n+2)>r
    text_file.write(str("%.4f \n" % F[2]))
    text_file.write('x1=')
    x1 = a + F[0] / F[2] * (b - a)
    text_file.write(str("%.4f \n" % x1))
    text_file.write('x2=')
    x2 = a + F[1] / F[2] * (b - a)
    text_file.write(str("%.4f \n" % x2))
    text_file.write('f(x1)=')
    y1 = fucntion(x1)
    text_file.write(str("%f \n" % y1))
    text_file.write('f(x2)=')
    y2 = fucntion(x2)
    text_file.write(str("%f \n" % y2))
    n -= 1
    text_file.write('number of iterations n+1:')
    text_file.write(str("%i \n" % n))
    while n > 1:
        text_file.write('f(x1)=')
        y1 = fucntion(x1)
        text_file.write(str("%f | " % y1))
        text_file.write('f(x2)=')
        y2 = fucntion(x2)
        text_file.write(str("%f \n" % y2))
        if y1 <= y2:
            b = x2
            x2 = x1
            x1 = a + (b - x2)
        else:
            a = x1
            x1 = x2
            x2 = b - (x1 - a)
        text_file.write('x1=')
        text_file.write(str("%.4f | " % x1))
        text_file.write('x2=')
        text_file.write(str("%.4f \n" % x2))
        n -= 1
        text_file.write(str("n="))
        text_file.write(str("%i \n" % n))
    text_file.write("x1=x2=")
    text_file.write(str("%.4f \n" % x1))
    text_file.write("f(x1;x2)=")
    text_file.write(str("%.4f \n" % y1))
    text_file.close()
    # функция в которой мы отделяем до разряда который входит в нашу заданную точность

    if toFixed(x1, 4) == toFixed(x2, 4):
        return toFixed(x1, 4)


fibonacci(0.0001, 0, 1)
