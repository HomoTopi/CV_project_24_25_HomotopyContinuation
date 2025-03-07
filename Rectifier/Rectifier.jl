include("../DataStructures/datastructures.jl")

module RectifierModule
    using ..HomographyModule
    using ..ConicsModule
    
    ##################################
    ### ---- exposed functions ----###
    ##################################
    function rectify(C_img::Conics)::Homography
        throw(ArgumentError("Abastract module, must be extended"))
    end

    export rectify
end
