using HomotopyContinuation

threshold = 1e-8

@var x y z

f_1 = a1 * x^2 + b1 * x * y + c1 * y^2 + d1 * x * z + e1 * y * z + f1 * z^2
f_2 = a2 * x^2 + b2 * x * y + c2 * y^2 + d2 * x * z + e2 * y * z + f2 * z^2
f_3 = a3 * x^2 + b3 * x * y + c3 * y^2 + d3 * x * z + e3 * y * z + f3 * z^2

F = System([f_1, f_2, f_3])
res = solve(F, [x, y, z]; show_progress=false)
sols = solutions(res; only_finite=true)

normalized = [sol / (norm(sol[3]) < threshold ? sol[1] : sol[3]) for sol in sols]

# Define a function that “rectifies” a complex number by setting its real or imaginary part to zero if that part is below the threshold.
rectify_component(z) = complex(
    abs(real(z)) < threshold ? zero(real(z)) : real(z),
    abs(imag(z)) < threshold ? zero(imag(z)) : imag(z)
)


complex_sols = [map(rectify_component, sol) for sol in normalized]

