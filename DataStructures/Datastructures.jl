module Datastructures
    import LinearAlgebra

    struct SceneDescription
        f::Float64
        theta::Float64
    end

    struct Conic
        M::Matrix{Float64} 

        function Conic(M::Matrix{Float64})
            if size(M) != (3, 3)
                throw(DimensionMismatch("Conic matrix must be 3×3, got $(size(M))"))
            end
            new(M)
        end
    end

    struct Conics
        C1::Conic
        C2::Conic
    end

    struct Homography
        H::Matrix{Float64}
    end

    struct Image
        h_true::Homography
        C_img::Conics
    end

    export SceneDescription, Conic, Conics, Homography, Image
end
