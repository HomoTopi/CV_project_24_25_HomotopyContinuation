using HomotopyContinuation

@var x y

f_1 = a1*x^2 + b1*x*y + c1*y^2 + d1*x + e1*y + f1

f_2 = a2*x^2 + b2*x*y + c2*y^2 + d2*x + e2*y + f2


F = System([f_1, f_2])

result = solutions(solve(F, [x, y]))
