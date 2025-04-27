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
    @var x[1:3], y[1:3]

    # Define the system of equations exactly as in rectify.jl
    f_1 = (x' * C_1 * x - y' * C_1 * y)^2 + 4 * (x' * C_1 * y)^2
    f_2 = (x' * C_2 * x - y' * C_2 * y)^2 + 4 * (x' * C_2 * y)^2
    f_3 = (x' * C_3 * x - y' * C_3 * y)^2 + 4 * (x' * C_3 * y)^2

    @info "f1 = " * string(f_1)
    @info "f2 = " * string(f_2)
    @info "f3 = " * string(f_3)

    f = f_1 + f_2 + f_3

    # Define constraint
    constraint = x' * x - y' * y
    @info "constraint = " * string(constraint)

    J_x = differentiate(f, x)
    J_y = differentiate(f, y)
    @info "J_x = " * string(J_x)
    @info "J_y = " * string(J_y)

    J_constraint_x = differentiate(constraint, x)
    J_constraint_y = differentiate(constraint, y)
    @info "J_constraint_x = " * string(J_constraint_x)

    @var lambda

    # Create system and solve
    # F = System([
    #         J_x[1] - lambda * J_constraint_x[1],
    #         J_x[2] - lambda * J_constraint_x[2],
    #         J_x[3] - lambda * J_constraint_x[3],
    #         J_y[1] - lambda * J_constraint_y[1],
    #         J_y[2] - lambda * J_constraint_y[2],
    #         J_y[3] - lambda * J_constraint_y[3],
    #         constraint
    #     ], variables=x ∪ y ∪ lambda)

    F = System([
            J_x[1],
            J_x[2],
            J_x[3],
            J_y[1],
            J_y[2],
            J_y[3],
            constraint
        ], variables=x ∪ y)


    @info "F = " * string(F)

    res = solve(F, [x, y, lambda]; show_progress=true)
    # F = System([df_dxi, df_dxr, df_dyi, df_dyr, df_dzi, df_dzr])
    # res = solve(F, [xr, xi, yr, yi, zr, zi]; show_progress=true)
    sols = solutions(res; only_finite=false, only_nonsingular=false, only_real=false)
    #@info "sols=" * string(sols)

    #Strip solutions of the lambda component
    if (length(sols) >= 6)
        sols = [sol[1:6] for sol in sols]
    end
    #@info "sols=" * string(sols)

    # Filter the solutions with imaginary parts less than the threshold
    sols = [real.(sol) for sol in sols if all(imag.(sol) .< threshold)]
    @info "filtered_sols=" * string(sols)

    # # Normalize solutions to make it norm 1
    # sols = [sol / norm(sol) for sol in sols]
    # @info "normalized_sols=" * string(sols)


    # # Select only real solutions
    # real_sols = [sol for sol in sols if all(isreal, sol)]
    # @info "real_sols=" * string(real_sols)

    # Evaluate f for all solutions, by letting x = sol[1:3], y = sol[4:6]
    evaluated_f = [Float64(
        subs(
            f,
            vcat(
                [(xi => vi) for (xi, vi) in zip(x, sol[1:3])],
                [(yi => vi) for (yi, vi) in zip(y, sol[4:6])]
            )
            ...
        )) for sol in sols]
    @info "evaluated_f=" * string(evaluated_f)

    # # Take the absolute value of the evaluated f (even if it is complex)
    # evaluated_f = [real(f_val * conj(f_val)) for f_val in evaluated_f]
    # @info "evaluated_f_abs=" * string(evaluated_f)


    # Find the two smallest solutions
    sorted_indices = sortperm(evaluated_f)
    smallest_indices = sorted_indices[1:2]
    smallest_solutions = [sols[i] for i in smallest_indices]
    @info "smallest_solutions=" * string(smallest_solutions)
    @info "smallest solutions values=" * string(evaluated_f[smallest_indices])

    # Define rectify_component function exactly as in rectify.jl
    rectify_component(z) = complex(
        abs(real(z)) < threshold ? zero(real(z)) : real(z),
        abs(imag(z)) < threshold ? zero(imag(z)) : imag(z)
    )

    #packet the solutions as complex numbers z = x_i + i y_i
    # where x_i and y_i are the ith components of the solution vector
    # and i is the imaginary unit

    smallest_solutions_packed = [[complex(sol[1], sol[4]), complex(sol[2], sol[5]), complex(sol[3], sol[6])] for sol in smallest_solutions]
    @info "smallest_solutions_packed=" * string(smallest_solutions_packed)

    #Normalize the smallest solutions by dividing by the first element of the solution vector
    smallest_solutions_packed = [sol / sol[1] for sol in smallest_solutions_packed]

    # Apply rectification to components
    complex_sols = [map(rectify_component, sol) for sol in smallest_solutions_packed]

    return complex_sols
end

# Example usage
c_1 = [1, 0.0, 1.0, 0.0, 0.0, -1.0]
c_2 = [1, 0.0, 1.0, -0.0, -1.0, -0.75]
c_3 = [1.0, 0.0, 1.0, 0.0, 0.0, -2.25]

params = vcat(c_1, c_2, c_3)
result = process_rectification(params)