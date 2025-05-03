using HTTP
using JSON3
using HomotopyContinuation
using LinearAlgebra

# Set threshold for numerical precision
const threshold = 1e-6

function process_rectification(params)
    #Base system
    c_1_base = [1, 0.0, 1.0, 0.0, 0.0, -1.0]
    c_2_base = [1, 0.0, 1.0, -0.0, -1.0, -0.75]
    c_3_base = [1.0, 0.0, 1.0, 0.0, 0.0, -2.25]

    C_1_base = [c_1_base[1] c_1_base[2]/2 c_1_base[4]/2; c_1_base[2]/2 c_1_base[3] c_1_base[5]/2; c_1_base[4]/2 c_1_base[5]/2 c_1_base[6]]
    @info "C_1_base = " * string(C_1_base)
    C_2_base = [c_2_base[1] c_2_base[2]/2 c_2_base[4]/2; c_2_base[2]/2 c_2_base[3] c_2_base[5]/2; c_2_base[4]/2 c_2_base[5]/2 c_2_base[6]]
    @info "C_2_base = " * string(C_2_base)
    C_3_base = [c_3_base[1] c_3_base[2]/2 c_3_base[4]/2; c_3_base[2]/2 c_3_base[3] c_3_base[5]/2; c_3_base[4]/2 c_3_base[5]/2 c_3_base[6]]
    @info "C_3_base = " * string(C_3_base)

    I = [1, 0, 0.0, 0, 1, 0.0, 0]
    #Add some small gaussian noise to the circular points
    I = I + randn(7) * 1e-6
    I = I / norm(I)
    J = [1, 0, 0.0, 0, -1, 0.0, 0]
    J = J / norm(J)
    #Add some small gaussian noise to the circular points
    J = J + randn(7) * 1e-6
    CircularPoints = vcat([I], [J])
    @info "CircularPoints = " * string(CircularPoints)

    # Extract parameters
    a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2, a3, b3, c3, d3, e3, f3 = params

    C_1_input = [a1 b1/2 d1/2; b1/2 c1 e1/2; d1/2 e1/2 f1]
    C_2_input = [a2 b2/2 d2/2; b2/2 c2 e2/2; d2/2 e2/2 f2]
    C_3_input = [a3 b3/2 d3/2; b3/2 c3 e3/2; d3/2 e3/2 f3]

    # Define variables - using same variable names as rectify.jl (x, y, z instead of x, y, w)
    @var x[1:3], y[1:3], C[1:3, 1:3], Cs[1:3, 1:3, 1:3], lambda

    # f(x,y) = |z^t C z|^2
    f = (x' * C * x - y' * C * y)^2 + 4 * (x' * C * y)^2
    @info "f = " * string(f)

    fSum = Expression(0.0)
    for i in 1:3
        fSum += subs(f, C => Cs[:, :, i])
    end
    @info "fSum = " * string(fSum)

    #|z|^2 = 1
    constraint = x' * x + y' * y - 1
    @info "constraint = " * string(constraint)

    #Objective function to be minimized
    J = fSum - lambda * constraint
    @info "J = " * string(J)

    # Compute the Jacobian
    dJ_dx = differentiate(J, x)
    dJ_dy = differentiate(J, y)
    dJ_dlambda = differentiate(J, lambda)
    @info "dJ_dx = " * string(dJ_dx)
    @info "dJ_dy = " * string(dJ_dy)
    @info "dJ_dlambda = " * string(dJ_dlambda)

    # Construct the System
    F = System(
        vcat(dJ_dx..., dJ_dy..., dJ_dlambda...),
        variables=vcat(x, y, [lambda]),
        parameters=vec(Cs)
    )

    # Correctly flatten and concatenate target parameters
    start_parameters = vcat(vec(C_1_base), vec(C_2_base), vec(C_3_base))
    target_parameters = vcat(vec(C_1_input), vec(C_2_input), vec(C_3_input))

    # # Compute the Jacobian matrix
    # jacobian_matrix = jacobian(F)
    # jacobian_matrix = subs(jacobian_matrix, Cs => reshape(target_parameters, size(Cs)))
    # @info "Jacobian matrix = " * string(jacobian_matrix)

    # # Compute the symbolic determinant of the Jacobian matrix
    # det_jacobian = det(jacobian_matrix)
    # #Evaluate at circular points
    # det_jacobian = [subs(det_jacobian, x => CircularPoints[i][1:3], y => CircularPoints[i][4:6], lambda => CircularPoints[i][7]) for i in 1:2]
    # @info "det_jacobian = " * string(det_jacobian)

    # return

    # Solve the system using homotopy continuation
    res = solve(F, [x, y, lambda], show_progress=true, start_system=:total_degree, target_parameters=target_parameters)
    # res = solve(F, CircularPoints, show_progress=true, start_parameters=start_parameters, target_parameters=target_parameters)
    @info "res = " * string(res)

    sols = solutions(res, only_real=true, only_finite=false, only_nonsingular=false)
    @info "sols = " * string(sols)

    # Find the two solutions with smallest evaluated value of J
    sols_evaluated = [subs(fSum, x => sol[1:3], y => sol[4:6], lambda => sol[7], Cs => reshape(target_parameters, size(Cs))) for sol in sols]
    # Keep only the real parts of the solutions
    sols_evaluated = [Float64(real(sol)) for sol in sols_evaluated]
    @info "sols_evaluated = " * string(sols_evaluated)

    rectify_component(z) = complex(
        abs(real(z)) < threshold ? zero(real(z)) : real(z),
        abs(imag(z)) < threshold ? zero(imag(z)) : imag(z)
    )

    # Find the indexes of the two smallest evaluated values of J
    sorted_indices = sortperm(sols_evaluated)
    @info "sorted_indices = " * string(sorted_indices)

    smallest_solution = sorted_indices[1]
    for s in sorted_indices
        if norm(smallest_solution - s) > threshold
            smallest_solution = [smallest_solution; s]
            break
        end
    end

    smallest_solutions = [real(sols[i]) for i in smallest_solution]
    @info "smallest_solutions = " * string(smallest_solutions)

    #Structure the solutions as complex numbers
    smallest_solutions = [sol[1:3] + im * sol[4:6] for sol in smallest_solutions]
    @info "smallest_solutions_complex = " * string(smallest_solutions)

    #Normalize the smallest solutions by dividing by the first element of the solution vector
    smallest_solutions = [sol / sol[1] for sol in smallest_solutions]
    @info "normalized_smallest_solutions = " * string(smallest_solutions)

    return smallest_solutions

end

# --- HTTP Server Setup ---
const ROUTER = HTTP.Router()

# Health check endpoint
HTTP.register!(ROUTER, "GET", "/health", req -> HTTP.Response(200, "OK"))

# Rectification endpoint that matches exactly what rectify.jl does
function handle_rectify(req::HTTP.Request)
    try
        # Parse JSON body
        json_body = JSON3.read(IOBuffer(req.body))

        # Expecting a structure like: {"conics": [c1_params, c2_params, c3_params]}
        # where each cX_params is [a, b, c, d, e, f]
        if !haskey(json_body, :conics) || length(json_body.conics) != 3 || !all(c -> length(c) == 6, json_body.conics)
            return HTTP.Response(400, JSON3.write((error = "Invalid input format")))
        end

        # Flatten the parameters
        params = vcat(Float64.(json_body.conics[1]), Float64.(json_body.conics[2]), Float64.(json_body.conics[3]))

        # Run the rectification process
        complex_sols = process_rectification(params)

        # Format response to match rectify.jl output
        response_body = JSON3.write((complex_sols=complex_sols,))
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=response_body)

    catch e
        return HTTP.Response(500, JSON3.write((error="Internal server error", details=sprint(showerror, e))))
    end
end

HTTP.register!(ROUTER, "POST", "/rectify", handle_rectify)

# --- Main Server Execution ---
@info "Starting Julia Rectifier server on 0.0.0.0:8081..."
HTTP.serve(ROUTER, "0.0.0.0", 8081)