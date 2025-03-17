module LosserModule
include("../DataStructures/Datastructures.jl")
using .Datastructures

##################################
### ---- exposed functions ----###
##################################
function computeLoss(H_true::Homography, H_computed::Homography)::float64
    throw(ArgumentError("Abastract module, must be extended"))
end

export computeLoss
end