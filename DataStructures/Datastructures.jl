module Datastructures
import LinearAlgebra

struct SceneDescription
    f::Float64
    theta::Float64
    circle1::Vector{Float64} # [centerX::Float64, centerY::Float64, radius::Float64]
    circle2::Vector{Float64}

    function SceneDescription(f::Float64, theta::Float64, circle1::Vector{Float64}, circle2::Vector{Float64})
        if (circle1[3] < 0 || circle2[3] < 0)
            throw(ArgumentError("Circle radius must be positive"))
        end
        new(f, theta, circle1, circle2)
    end
end

struct Conic
    M::Matrix{Float64}

    function Conic(M::Matrix{Float64})
        if size(M) != (3, 3)
            throw(DimensionMismatch("Conic matrix must be 3×3, got $(size(M))"))
        end
        if !isapprox(M, M')
            throw(ArgumentError("Conic matrix must be symmetric"))
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

struct Img
    h_true::Homography
    C_img::Conics
end

export SceneDescription, Conic, Conics, Homography, Img
end
