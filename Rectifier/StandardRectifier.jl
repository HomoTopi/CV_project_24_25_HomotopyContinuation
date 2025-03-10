module StandardRectifier
    include("Rectifier.jl")
    import .RectifierModule: rectify, Conics, Homography, Conic, conic_par2alg
    #using NonlinearSolve
    using SymPy
    using LinearAlgebra
    # Extend the function from Main.RectifierModule using its absolute qualified name
    function rectify(C_img::Conics)::Homography
        @info "Rectifying using SymPy"

        @debug "Conics: $C_img"
        a1, b1, c1, d1, e1, f1 = conic_par2alg(C_img.C1)
        a2, b2, c2, d2, e2, f2 = conic_par2alg(C_img.C2)

        @info "Equation 1: $a1*x^2 + $b1*x*y + $c1*y^2 + $d1*x + $e1*y + $f1"
        @info "Equation 2: $a2*x^2 + $b2*x*y + $c2*y^2 + $d2*x + $e2*y + $f2"

        @syms x y

        eq1 = Eq(a1*x^2 + b1*x*y + c1*y^2 + d1*x + e1*y + f1, 0)
        eq2 = Eq(a2*x^2 + b2*x*y + c2*y^2 + d2*x + e2*y + f2, 0)

        solutions = solve((eq1, eq2), (x, y))
        @info "Solutions: $solutions"
        II = [N(solutions[1][1]); N(solutions[1][2])]
        JJ = [N(solutions[2][1]); N(solutions[2][2])]
        @debug "II: $II"
        @debug "JJ: $JJ"

        imDCCP = II*JJ' + JJ*II'

        imDCCP = imDCCP./norm(imDCCP)
        @debug "imDCCP: $imDCCP"

        # extract the line at the infinity (2fix)
        #l_inf = nullspace(imDCCP)
        #@info "l_inf: $l_inf"

        #H_1 = Homography([I(2) zeros(2,1); l_inf])
        #@info "H_1: $H_1"


        U, S, V = svd(imDCCP)
        @info "U: $U"
        @info "S: $S"
        @info "V: $V"

        t =  Diagonal(1 ./ sqrt.(S)) * U'

        H_2 = [t zeros(2,1); zeros(1,2) 1]
        @info "H_2: $H_2"

        #H = H_1*H_2
        #@info "H: $H"
        
        # Do something with the solution
        # For now, just return a random homography
        return Homography(rand(3,3))
    end
end