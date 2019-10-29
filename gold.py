import math


def gold(e, a, b):
    file = open('../output/gold.txt', 'w')
    i = 1
    a = 0
    b = 1
    x1 = a + 0.381966011 * (b - a)
    x2 = a + 0.618003399 * (b - a)

    def func(x):
        res = math.pow(x, 3) - x
        return res

    f1 = func(x1)
    f2 = func(x2)

    def next_iter(i, e, a, b, x1, x2, f1, f2, file):
        file.write(
            "Iteration " + str(i) + ":\nx1 = " + str(x1) + "; x2 = " + str(x2) + ";\nf1 = " + str(f1) + "; f2 = " + str(
                f2) + ";\n\n")
        i += 1
        if f1 < f2:
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

        if abs(b - a) < e:
            file.write("Iteration " + str(i) + ":\nx1 = " + str(x1) + "; x2 = " + str(x2) + ";\nf1 = " + str(
                f1) + "; f2 = " + str(f2) + ";\n\n")
            answer = (x1 + x2) / 2
            file.write("\nANSWER: " + str(answer))
            file.close()
            return answer
        else:
            return next_iter(i, e, a, b, x1, x2, f1, f2, file)

    return next_iter(i, e, a, b, x1, x2, f1, f2, file)
