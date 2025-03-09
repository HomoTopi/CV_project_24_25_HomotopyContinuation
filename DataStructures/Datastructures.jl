module Datastructures
    import LinearAlgebra

    struct SceneDescription
        f::Float64
        theta::Float64
    end

    struct Conics
        C1::Matrix{Float64}
        C2::Matrix{Float64}
    end

    struct Homography
        H::Matrix{Float64}
    end

    struct Image
        h_true::Homography
        C_img::Conics
    end

    export SceneDescription, Conics, Homography, Image
end
