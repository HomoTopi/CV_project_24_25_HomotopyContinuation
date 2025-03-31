using HomotopyContinuation

@var x y w

f_1 = a1*x^2 + b1*x*y + c1*y^2 + d1*x*w + e1*y*w + f1*w^2
f_2 = a2*x^2 + b2*x*y + c2*y^2 + d2*x*w + e2*y*w + f2*w^2
f_3 = a3*x^2 + b3*x*y + c3*y^2 + d3*x*w + e3*y*w + f3*w^2

F = System([f_1, f_2, f_3])
res = solve(F, [x, y, w])
sol = solutions(res)
#real_sol = real_solutions(res)
