module ImageWarperModule
include("../DataStructures/Datastructures.jl")
using .Datastructures
using Images

##################################
### ---- exposed functions ----###
##################################
function warpImage(src_img::Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}}, H::Homography)::Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}}
    img_width, img_height = size(src_img)
    warped_img = zeros(RGB{N0f8}, img_width, img_height)

    H_inv = inv(H.H)

    for x = 1:img_width
        for y = 1:img_height
            p = H_inv * [x; y; 1]
            p = p / p[3]
            x_new = round(Int, p[1])
            y_new = round(Int, p[2])

            if 1 <= x_new <= img_width && 1 <= y_new <= img_height
                warped_img[y, x] = src_img[y_new, x_new]
            end
        end
    end

    return warped_img
end

export warpImage
end
