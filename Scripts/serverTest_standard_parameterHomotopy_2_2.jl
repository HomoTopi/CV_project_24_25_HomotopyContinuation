using HomotopyContinuation
using LinearAlgebra

# Set threshold for numerical precision
const threshold = 1e-8

function process_rectification(params)
    #Base system
    c_1_base = [1, 0.0, 1.0, 0.0, 0.0, -1.0]
    c_2_base = [1, 0.0, 1.0, -0.0, -1.0, -0.75]
    c_3_base = [1.0, 0.0, 1.0, 0.0, 0.0, -2.25]

    C_1_base = [c_1_base[1] c_1_base[2]/2 c_1_base[4]/2; c_1_base[2]/2 c_1_base[3] c_1_base[5]/2; c_1_base[4]/2 c_1_base[5]/2 c_1_base[6]]
    C_2_base = [c_2_base[1] c_2_base[2]/2 c_2_base[4]/2; c_2_base[2]/2 c_2_base[3] c_2_base[5]/2; c_2_base[4]/2 c_2_base[5]/2 c_2_base[6]]

    I = [1, im, 0.0]
    J = [1, -im, 0.0]
    CircularPoints = vcat([I], [J])
    @info "CircularPoints = " * string(CircularPoints)

    # Extract parameters
    a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2, a3, b3, c3, d3, e3, f3 = params

    C_1_input = [a1 b1/2 d1/2; b1/2 c1 e1/2; d1/2 e1/2 f1]
    C_2_input = [a2 b2/2 d2/2; b2/2 c2 e2/2; d2/2 e2/2 f2]
    C_3_input = [a3 b3/2 d3/2; b3/2 c3 e3/2; d3/2 e3/2 f3]

    # Define variables - using same variable names as rectify.jl (x, y, z instead of x, y, w)
    @var x[1:3], C_1[1:3, 1:3], C_2[1:3, 1:3], C_3[1:3, 1:3]

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
    F = System([f_1, f_2], variables=x, parameters=vec(C_1) ∪ vec(C_2))
    @info "F = " * string(F)

    function solveSystem(C_1_start, C_2_start, C_1_target, C_2_target, currentSol)
        res_ = solve(F, currentSol,
            start_parameters=Float64.(vcat(vec(C_1_start), vec(C_2_start))),
            target_parameters=Float64.(vcat(vec(C_1_target), vec(C_2_target)));
            show_progress=true)
        sols = solutions(res_; only_finite=false, only_nonsingular=true)
        #Normalize the solutions by dividing by the first element of the solution vector
        sols = [sol / sol[1] for sol in sols]
        return sols
    end

    sols_12 = solveSystem(C_1_base, C_2_base, C_1_input, C_2_input, CircularPoints)
    @info "sols_12=" * string(sols_12)
    sols_13 = solveSystem(C_1_input, C_2_input, C_1_input, C_3_input, sols_12)
    @info "sols_13=" * string(sols_13)
    sols_23 = solveSystem(C_1_input, C_2_input, C_3_input, C_2_input, sols_12)
    @info "sols_23=" * string(sols_23)

    #Compute the average of the solutions
    sols = [(sol_12 + sol_13 + sol_23) / 3 for (sol_12, sol_13, sol_23) in zip(sols_12, sols_13, sols_23)]
    @info "sols=" * string(sols)

    # Define rectify_component function exactly as in rectify.jl
    rectify_component(z) = complex(
        abs(real(z)) < threshold ? zero(real(z)) : real(z),
        abs(imag(z)) < threshold ? zero(imag(z)) : imag(z)
    )

    #Normalize the smallest solutions by dividing by the first element of the solution vector
    sols = [sol / sol[1] for sol in sols]

    # Apply rectification to components
    complex_sols = [map(rectify_component, sol) for sol in sols]

    return complex_sols
end

# Example usage
c_1 = [0.98, 0.0, 1.0, 0.0, 1.0, -1.0]
c_2 = [1, 0.0, 1, -0.0, -0.8, -0.75]
c_3 = [1.0, 0.0, 1.0, 0.0, 0.0, -2.25]

params = vcat(c_1, c_2, c_3)
result = process_rectification(params)