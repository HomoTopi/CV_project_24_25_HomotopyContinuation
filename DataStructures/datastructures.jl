module SceneDescriptionModule
    struct SceneDescription
        f::Float64
        theta::Float64
    end

    export SceneDescription
end

module ConicsModule
    import LinearAlgebra
    struct Conics
        C1::Matrix{Float64}
        C2::Matrix{Float64}
    end

    export Conics
end

module HomographyModule
    import LinearAlgebra
    struct Homography
        H::Matrix{Float64}
    end
    
    export Homography
end


module ImageModule
    using ..HomographyModule
    using ..ConicsModule
    struct Image
        h_true::Homography
        C_img::Conics
    end
    
    export Image
end