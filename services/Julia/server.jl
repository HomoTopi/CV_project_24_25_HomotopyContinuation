using HTTP
using JSON3
using HomotopyContinuation
using LinearAlgebra

# Set threshold for numerical precision
const threshold = 1e-8

function process_rectification(params)
    # Extract parameters
    a1, b1, c1, d1, e1, f1, a2, b2, c2, d2, e2, f2, a3, b3, c3, d3, e3, f3 = params
    
    # Define variables - using same variable names as rectify.jl (x, y, z instead of x, y, w)
    @var x y z
    
    # Define the system of equations exactly as in rectify.jl
    f_1 = a1 * x^2 + b1 * x * y + c1 * y^2 + d1 * x * z + e1 * y * z + f1 * z^2
    f_2 = a2 * x^2 + b2 * x * y + c2 * y^2 + d2 * x * z + e2 * y * z + f2 * z^2
    f_3 = a3 * x^2 + b3 * x * y + c3 * y^2 + d3 * x * z + e3 * y * z + f3 * z^2
    

    @info "f1=" * string(f_1)
    @info "f2=" * string(f_2)
    @info "f3=" * string(f_3)
    # Create system and solve
    F = System([f_1, f_2, f_3])
    res = solve(F, [x, y, z]; show_progress=true)
    sols = solutions(res; only_finite=true)
    @info "sols=" * string(sols)
    
    # Normalize solutions exactly as in rectify.jl
    normalized = [sol / (norm(sol[3]) < threshold ? sol[1] : sol[3]) for sol in sols]
    
    # Define rectify_component function exactly as in rectify.jl
    rectify_component(z) = complex(
        abs(real(z)) < threshold ? zero(real(z)) : real(z),
        abs(imag(z)) < threshold ? zero(imag(z)) : imag(z)
    )
    
    # Apply rectification to components
    complex_sols = [map(rectify_component, sol) for sol in normalized]
    
    return complex_sols
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
            return HTTP.Response(400, JSON3.write((error="Invalid input format")))
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