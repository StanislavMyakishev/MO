import math


def fucntion(x):
    res = math.pow(x, 3)-x
    return res


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


def fibonacci(e, a, b):
    F = list()  # список наши Fn , Fn+1 , Fn+2
    result = 0
    r = (b-a)/e
    n = 1
    znach = math.pow((((1 + math.sqrt(5))/2)), n) / math.sqrt(5)
    while znach < r:
        n += 1
        znach = math.pow((((1 + math.sqrt(5))/2)), n) / math.sqrt(5)

    F.append(math.pow((((1 + math.sqrt(5))/2)), n-2) / math.sqrt(5))  # F(n)
    F.append(math.pow((((1 + math.sqrt(5))/2)), n-1) / math.sqrt(5))  # F(n+1)
    F.append(znach)  # F(n+2)>r
    x1 = a+F[0]/F[2]*(b-a)
    x2 = a+F[1]/F[2]*(b-a)

    y1 = fucntion(x1)
    y2 = fucntion(x2)

    n -= 1
    while n > 1:
        y1 = fucntion(x1)
        y2 = fucntion(x2)

        if (y1 <= y2):
            b = x2
            x2 = x1
            x1 = a+(b-x2)
        else:
            a = x1
            x1 = x2
            x2 = b-(x1-a)
        n -= 1

    # функция в которой мы отделяем до разряда который входит в нашу заданную точность
    if(toFixed(x1, 4) == toFixed(x2, 4)):
        return toFixed(x1, 4)


print(fibonacci(0.0001, 0, 1))
