#Reset environment
using Pkg
Pkg.activate("..")

# Load modules
include("ImageWarper.jl")

import .ImageWarperModule: warpImage, Homography
using Images
using Test

img = load("./TestImages/Lena.png")
img = convert(Array{RGB{N0f8},2}, img)
H = Homography(
    [
        1.0 0.1 20.0;
        0.0 1.0 20.0;
        0.001 0.0 1.0
    ]
)

warped_img = warpImage(img, H)

@test size(warped_img) == size(img)
@test typeof(warped_img) == Array{RGB{N0f8},2}

# plot original and warped image
mosaicview([img warped_img])


