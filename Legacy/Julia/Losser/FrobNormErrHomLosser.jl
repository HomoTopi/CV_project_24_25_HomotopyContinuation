module FrobNormErrHomLosserModule
include("./Losser.jl")

import .LosserModule: computeLoss, Homography
import LinearAlgebra

#Extends the computeLoss function from LosserModule
function computeLoss(H_true::Homography, H_computed::Homography)::Float64
    H_computed_inv = LinearAlgebra.inv(H_computed.H)
    return LinearAlgebra.norm(H_computed_inv * H_true.H - ones(3, 3))
end
end