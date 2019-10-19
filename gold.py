import math

def gold(e, a, b):

    file = open('gold_output.txt', 'w')
    i = 1
    a = 0
    b = 1
    x1 = a + 0.381966011 * (b - a)
    x2 = a + 0.618003399 * (b - a)

    def func(x):
        res = math.pow(x, 3)-x
        return res

    f1 = func(x1)
    f2 = func(x2)

    def next(i, e, a, b, x1, x2, f1, f2, file):
        file.write("Iteration " + str(i) + ":\nx1 = " + str(x1) + "; x2 = " + str(x2) + ";\na = " + str(a) + "; b = " + str(b) + ";\n\n")
        i += 1
        if (f1 < f2):
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + 0.381966011 * (b - a)
            f1 = func(x1)
            print(x1, x2)

        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + 0.618003399 * (b - a)
            f2 = func(f2)
            print(x1, x2)

        if (abs(b - a) < e):
            file.write("Iteration " + str(i) + ":\nx1 = " + str(x1) + "; x2 = " + str(x2) + ";\na = " + str(a) + "; b = " + str(b) + ";\n\n")
            answer = (x1 + x2) / 2
            file.write("\nANSWER: " + str(answer))
            file.close()
            return answer
        else:
            return next(i, e, a, b, x1, x2, f1, f2, file)

    return next(i, e, a, b, x1, x2, f1, f2, file)
    
print(gold(0.0001, 0, 1))

