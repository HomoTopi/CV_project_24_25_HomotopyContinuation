using HomotopyContinuation
using LinearAlgebra

# Set threshold for numerical precision
const threshold = 1e-8

function matrixFromEllipseParams(a, b, x_0, y_0, costheta, sintheta)
    A = a^2 * sintheta^2 + b^2 * costheta^2
    B = 2 * (b^2 - a^2) * costheta * sintheta
    C = a^2 * costheta^2 + b^2 * sintheta^2
    D = -2 * A * x_0 - B * y_0
    E = -B * x_0 - 2 * C * y_0
    F = A * x_0^2 + B * x_0 * y_0 + C * y_0^2 - a^2 * b^2
    # Create the matrix from ellipse parameters
    return [
        A B/2 D/2;
        B/2 C E/2;
        D/2 E/2 F
    ]
end

function EllipseParamsFromMatrixParams(A, B, C, D, E, F)
    tmp = sqrt((A - C)^2 + B^2)
    a = -sqrt(2(A * E^2 + C * D^2 - B * D * E + (B^2 - 4 * A * C) * F) * (A + C + tmp)) / (B^2 - 4 * A * C)
    b = -sqrt(2(A * E^2 + C * D^2 - B * D * E + (B^2 - 4 * A * C) * F) * (A + C - tmp)) / (B^2 - 4 * A * C)
    x_0 = (2 * C * D - B * E) / (B^2 - 4 * A * C)
    y_0 = (2 * A * E - B * D) / (B^2 - 4 * A * C)
    theta = 0.5 * atan(-B, C - A)
    costheta = cos(theta)
    sintheta = sin(theta)

    return [a, b, x_0, y_0, costheta, sintheta]
end

function process_rectification(params)
    I = [1, im, 0.0]
    J = [1, -im, 0.0]
    CircularPoints = vcat([I], [J])

    #Base system
    c_1_base = [1, 1, 0.0, 0.0, 1.0, 0.0]
    c_2_base = [1, 1, 1.0, 0.0, 1.0, 0.0]
    c_3_base = [1.5, 1.5, 0.0, 0.0, 1.0, 0.0]

    baseParams = reshape(vcat(c_1_base', c_2_base', c_3_base'), :, 6)

    # Extract parameters
    a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2, a3, b3, c3, d3, e3, f3 = params

    c_1_params_input = EllipseParamsFromMatrixParams(a1, b1, c1, d1, e1, f1)
    c_2_params_input = EllipseParamsFromMatrixParams(a2, b2, c2, d2, e2, f2)
    c_3_params_input = EllipseParamsFromMatrixParams(a3, b3, c3, d3, e3, f3)

    inputParams = reshape(vcat(c_1_params_input', c_2_params_input', c_3_params_input'), :, 6)
    @info "ellipsisParams = " * string(ellipsisParams)

    # Define variables - using same variable names as rectify.jl (x, y, z instead of x, y, w)
    @var x[1:3], ellipsisParams[1:3, 1:6], ellipseParams[1:6]

    # Define the system of equations exactly as in rectify.jl
    C = matrixFromEllipseParams(ellipseParams[1], ellipseParams[2], ellipseParams[3], ellipseParams[4], ellipseParams[5], ellipseParams[6])
    @info "C = " * string(C)
    f = x' * C * x
    @info "f = " * string(f)

    supF = subs(f, ellipseParams => ellipsisParams[1, :])
    @info "sup f = " * string(supF)

    thetaConstraint = ellipseParams[5]^2 + ellipseParams[6]^2 - 1
    @info "thetaConstraint = " * string(thetaConstraint)


    # Create system and solve
    fs = [subs(f, ellipseParams => param) for param in eachrow(ellipsisParams)]
    constraints = [subs(thetaConstraint, ellipseParams => param) for param in eachrow(ellipsisParams)]

    equations = cat(fs, constraints; dims=1)
    @info "equations = " * string(equations)

    F = System(equations, variables=x, parameters=vec(ellipsisParams))
    @info "F = " * string(F)

    @info "sizeBaseparams=" * string(size(baseParams))
    @info "sixeellipsisParams=" * string(size(ellipsisParams))

    # Check if the circular points satisfy the system
    circular_point_solutions = [subs(equations, x => point) for point in CircularPoints]
    circular_point_solutions = [subs(sol, ellipsisParams => baseParams) for sol in circular_point_solutions]
    @info "Circular point solutions = " * string(circular_point_solutions)
    # res = solve(F, [x]; show_progress=true, target_parameters=vec(baseParams))
    # # F = System([df_dxi, df_dxr, df_dyi, df_dyr, df_dzi, df_dzr])
    # # res = solve(F, [xr, xi, yr, yi, zr, zi]; show_progress=true)
    # sols = solutions(res; only_finite=false, only_nonsingular=true)
    # @info "sols_base=" * string(sols)


    res = solve(F, CircularPoints,
        start_parameters=vec(baseParams),
        target_parameters=vec(inputParams);
        show_progress=true)
    @info "res=" * string(res)

    sols = solutions(res; only_finite=false, only_nonsingular=true)
    @info "sols_input=" * string(sols)

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
c_1 = [1, 0.0, 1.0, 0.0, 0.0, -1.0]
c_2 = [0.9, 0.0, 1, -0.0, -1.0, -0.75]
c_3 = [1.0, 0.0, 1.0, 0.0, 0.0, -2.25]

params = vcat(c_1, c_2, c_3)
result = process_rectification(params)