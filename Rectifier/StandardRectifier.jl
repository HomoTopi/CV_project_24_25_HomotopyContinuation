module StandardRectifier
    include("Rectifier.jl")
    import .RectifierModule: rectify, Conics, Homography, Conic, conic_par2alg
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
        @debug "Solutions: $solutions"
        II = [N(solutions[1][1]); N(solutions[1][2]); 1]
        JJ = [N(solutions[2][1]); N(solutions[2][2]); 1]
        @debug "II: $II"
        @debug "JJ: $JJ"

        imDCCP = II*JJ' + JJ*II'

        imDCCP = imDCCP./norm(imDCCP)
        @debug "imDCCP: $imDCCP"


        U, S, V = svd(imDCCP)
        @debug "U: $U"
        @debug "S: $S"
        @debug "V: $V"

        H =  Diagonal(1 ./ sqrt.(S)) * U'

        @debug "H: $H"

        return Homography(H)
    end
end