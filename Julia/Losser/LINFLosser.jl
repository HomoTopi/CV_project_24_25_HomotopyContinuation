module LINFLosserModule
include("./Losser.jl")
import .LosserModule: computeLoss, Homography
import LinearAlgebra

#Extends the computeLoss function from LosserModule
function computeLoss(H_true::Homography, H_computed::Homography)::Float64
    img_l_inf_true = H_true.H[3, :]
    img_l_inf_true = img_l_inf_true / img_l_inf_true[3]

    img_l_inf_computed = H_computed.H[3, :]
    img_l_inf_computed = img_l_inf_computed / img_l_inf_computed[3]

    #Return the angle between the two lines by computing the dot product
    return LinearAlgebra.dot(img_l_inf_true, img_l_inf_computed) / (LinearAlgebra.norm(img_l_inf_true) * LinearAlgebra.norm(img_l_inf_computed))
end
end