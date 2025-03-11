module ReprErrLosserModule
include("./Losser.jl")

import .LosserModule: Homography
import LinearAlgebra

#Extends the computeLoss function from LosserModule
function computeLoss(H_true::Homography, H_computed::Homography, width::Int, height::Int)::Float64
    corners = [
        0 width 0 width;
        0 0 height height;
        1 1 1 1
    ]

    corners_true = H_true.H * corners
    corners_true = corners_true ./ transpose(corners_true[3, :])

    corners_computed = H_computed.H * corners
    corners_computed = corners_computed ./ transpose(corners_computed[3, :])

    return LinearAlgebra.norm(corners_true - corners_computed)
end

export computeLoss
end