module RectifierModule
    include("../DataStructures/Datastructures.jl")
    import .HomographyModule: Homography
    import .ConicsModule: Conics
    
    ##################################
    ### ---- exposed functions ----###
    ##################################
    function rectify(C_img::Conics)::Homography
        throw(ArgumentError("Abastract module, must be extended"))
    end

    export rectify
end
