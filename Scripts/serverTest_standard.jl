using HomotopyContinuation
using LinearAlgebra

# Set threshold for numerical precision
const threshold = 1e-8

function process_rectification(params)
    # Extract parameters
    a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2, a3, b3, c3, d3, e3, f3 = params

    C_1 = [a1 b1/2 d1/2; b1/2 c1 e1/2; d1/2 e1/2 f1]
    C_2 = [a2 b2/2 d2/2; b2/2 c2 e2/2; d2/2 e2/2 f2]
    C_3 = [a3 b3/2 d3/2; b3/2 c3 e3/2; d3/2 e3/2 f3]

    # Define variables - using same variable names as rectify.jl (x, y, z instead of x, y, w)
    @var x[1:3]

    # Define the system of equations exactly as in rectify.jl
    f_1 = x' * C_1 * x
    f_2 = x' * C_2 * x
    f_3 = x' * C_3 * x

    @info "f1 = " * string(f_1)
    @info "f2 = " * string(f_2)
    @info "f3 = " * string(f_3)

    # Define constraint
    constraint = x' * x
    @info "constraint = " * string(constraint)

    # Create system and solve
    F = System([f_1, f_2, f_3], variables=x)
    @info "F = " * string(F)

    res = solve(F, [x]; show_progress=true)
    # F = System([df_dxi, df_dxr, df_dyi, df_dyr, df_dzi, df_dzr])
    # res = solve(F, [xr, xi, yr, yi, zr, zi]; show_progress=true)
    sols = solutions(res; only_finite=false, only_nonsingular=false)
    @info "sols=" * string(sols)

    #Evaluate the system at the solutions
    evaluated_solutions = [subs([f_1, f_2, f_3], x => sol) for sol in sols]
    evaluated_solutions = [[norm(Complex{Float64}(sol_)) for sol_ in sol] for sol in evaluated_solutions]
    evaluated_solutions = [sum(sol) for sol in evaluated_solutions]
    @info "evaluated_solutions=" * string(evaluated_solutions)

    #Take the two sollutions with the smallest evaluated_solutions
    smallest_indices = sortperm(evaluated_solutions)[1:2]
    @info "smallest_indices=" * string(smallest_indices)
    smallest_solutions = [sols[i] for i in smallest_indices]
    @info "smallest_solutions=" * string(smallest_solutions)

    # Define rectify_component function exactly as in rectify.jl
    rectify_component(z) = complex(
        abs(real(z)) < threshold ? zero(real(z)) : real(z),
        abs(imag(z)) < threshold ? zero(imag(z)) : imag(z)
    )

    #Normalize the smallest solutions by dividing by the first element of the solution vector
    smallest_solutions = [sol / sol[1] for sol in smallest_solutions]

    # Apply rectification to components
    complex_sols = [map(rectify_component, sol) for sol in smallest_solutions]
    @info "complex_sols=" * string(complex_sols)
    return complex_sols
end

# Example usage
c_1 = [1, 0.0, 1.0, 0.0, 0.0, -1.0]
c_2 = [1, 0.0, 1, -0.0, -1.0, -0.75]
c_3 = [1.0, 0.0, 1.0, 0.0, 0.0, -2.25]

params = vcat(c_1, c_2, c_3)
result = process_rectification(params)