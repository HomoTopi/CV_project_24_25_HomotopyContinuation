module FrobNormLosserModule
include("./Losser.jl")

import .LosserModule: computeLoss, Homography
import LinearAlgebra

#Extends the computeLoss function from LosserModule
function computeLoss(H_true::Homography, H_computed::Homography)::Float64
    return LinearAlgebra.norm(H_true.H - H_computed.H)
end
end