module RectifierModule
    include("../DataStructures/Datastructures.jl")
    using .Datastructures
    
    ##################################
    ### ---- exposed functions ----###
    ##################################
    function rectify(C_img::Conics)::Homography
        throw(ArgumentError("Abastract module, must be extended"))
    end

    function conic_par2alg(C::Conic)
        ## Conversion from the matrix form to the parameters form
        #a = C[1,1]
        #b = C[1,2]*2
        #c = C[2,2]
        #d = C[1,3]*2
        #e = C[2,3]*2
        #f = C[3,3]
        #return a, b, c, d, e, f
        return C.M[1,1], C.M[1,2]*2, C.M[2,2], C.M[1,3]*2, C.M[2,3]*2, C.M[3,3]
    end

    export rectify, conic_par2alg
end
