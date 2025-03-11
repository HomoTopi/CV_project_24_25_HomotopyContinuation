include("FrobNormLosser.jl")
include("../ImageWarper/ImageWarper.jl")

import .FrobNormLosserModule: computeLoss, Homography
import Images
using Plots

#load an image
img = Images.load("./TestImages/Lena.png")
img = convert(Array{Images.RGB{Images.N0f8},2}, img)

H_true = Homography(
    [1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0]
)

# Compute random gaussian noise matrix
mu = 0
sigma = 0.001
eps = randn(3, 3) .* sigma .+ mu

H_computed = Homography(H_true.H + eps)


println("H_true: $(H_true.H)")
println("H_computed: $(H_computed.H)")
println("Loss: $(computeLoss(H_true, H_computed))")

# Warp the image
warped_img_true = ImageWarperModule.warpImage(img, ImageWarperModule.Homography(H_true.H))
warped_img_computed = ImageWarperModule.warpImage(img, ImageWarperModule.Homography(H_computed.H))

# Plot the images as plots with titles
p1 = heatmap(img, title="Original", axis=false)
p2 = heatmap(warped_img_true, title="Warped True", axis=false)
p3 = heatmap(warped_img_computed, title="Warped Computed", axis=false)

img_width, img_height = size(img)
plot(p1, p2, p3, layout=(3, 1), size=(img_width, img_height * 3), fmt=:png)
