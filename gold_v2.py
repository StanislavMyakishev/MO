import math
from math import sqrt

gr = (1 + sqrt(5)) / 2

def golden_section(f, a, b, tolerance=1e-4, max_iterations=100):
    
    file = open('gold_output.txt', 'w')
    iteration = 0
    x1 = b - (b - a) / gr
    x2 = a + (b - a) / gr

    while iteration < max_iterations:
        
        file.write("Iteration " + str(iteration) + ":\na = " + str(a) + "; b = " + str(b) + ";\n")
        file.write("x1 = " + str(x1) + "; x2 = " + str(x2) + ";\n")
        file.write("f(x1) = " + str(f(x1)) + "; f(x2) = " + str(f(x2)) + ";\n\n")
        iteration += 1

        if abs(x1 - x2) < tolerance:
            return (b + a) / 2

        if f(x1) < f(x2):
            b = x2
        else:
            a = x1

        # we recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        x1 = b - (b - a) / gr
        x2 = a + (b - a) / gr
        
    file.close()
    return 'Max iterations exceeded'

f = lambda x: (x**3 - x)

print(golden_section(f, 0, 1))