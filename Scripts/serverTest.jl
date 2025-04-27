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

    @info "real(f1) = " * string(real(f_1))
    @info "imag(f1) = " * string(imag(f_1))

    f = real(f_1)^2 + real(f_2)^2 + real(f_3)^2 + imag(f_1)^2 + imag(f_2)^2 + imag(f_3)^2
    @info "f = " * string(f)

    # Define constraint
    constraint = x' * x - 1
    @info "constraint = " * string(constraint)

    J = differentiate(f, x)
    @info "J = " * string(J)

    # Create system and solve
    F = System([J[1], J[2], J[3]], variables=x)
    @info "F = " * string(F)

    res = solve(F, [x]; show_progress=true)
    # F = System([df_dxi, df_dxr, df_dyi, df_dyr, df_dzi, df_dzr])
    # res = solve(F, [xr, xi, yr, yi, zr, zi]; show_progress=true)
    sols = solutions(res; only_finite=false, only_nonsingular=false)
    @info "sols=" * string(sols)

    # Normalize solutions to make it norm 1
    normalized_sols = [sol / norm(sol) for sol in sols]
    @info "normalized_sols=" * string(normalized_sols)

    # Handle case where no solutions are found
    if isempty(normalized_sols)
        @warn "No solutions found"
        return []
    end

    # # Select only real solutions
    # real_sols = [sol for sol in sols if all(isreal, sol)]
    # @info "real_sols=" * string(real_sols)

    # Evaluate f for all solutions, by letting x = sol[1], y = sol[2], z = sol[3]
    evaluated_f = [Complex{Float64}(subs(f, x => sol)) for sol in normalized_sols]
    @info "evaluated_f=" * string(evaluated_f)

    # Take the absolute value of the evaluated f (even if it is complex)
    evaluated_f = [real(f_val * conj(f_val)) for f_val in evaluated_f]
    @info "evaluated_f_abs=" * string(evaluated_f)


    # Find the two smallest solutions
    sorted_indices = sortperm(evaluated_f)
    smallest_indices = sorted_indices[1:2]
    smallest_solutions = [normalized_sols[i] for i in smallest_indices]
    @info "smallest_solutions=" * string(smallest_solutions)
    @info "smallest solutions values=" * string(evaluated_f[smallest_indices])

    # Define rectify_component function exactly as in rectify.jl
    rectify_component(z) = complex(
        abs(real(z)) < threshold ? zero(real(z)) : real(z),
        abs(imag(z)) < threshold ? zero(imag(z)) : imag(z)
    )

    #Normalize the smallest solutions by dividing by the first element of the solution vector
    smallest_solutions = [sol / sol[1] for sol in smallest_solutions]

    # Apply rectification to components
    complex_sols = [map(rectify_component, sol) for sol in smallest_solutions]

    return complex_sols
end

# Example usage
c_1 = [1, 0.0, 1.0, 0.0, 0.0, -1.0]
c_2 = [1, 0.0, 1.0, -0.0, -1.0, -0.75]
c_3 = [1.0, 0.0, 1.0, 0.0, 0.0, -2.25]

params = vcat(c_1, c_2, c_3)
result = process_rectification(params)