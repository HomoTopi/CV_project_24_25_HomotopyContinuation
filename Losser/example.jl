#Reset environment
using Pkg
Pkg.activate("..")

include("FrobNormLosser.jl")
include("LINFLosser.jl")
include("ReprErrLosser.jl")
include("FrobNormErrHomLosser.jl")
include("SKSLosser.jl")
include("../ImageWarper/ImageWarper.jl")

import .FrobNormLosserModule: computeLoss, Homography
import .LINFLosserModule: computeLoss as LINFLosserComputeLoss, Homography as LINFLosserHomography
import .ReprErrLosserModule: computeLoss as ReprErrLossrComputeLoss, Homography as ReprErrLossrHomography
import .FrobNormErrHomLosserModule: computeLoss as FrobNormErrHomLosserComputeLoss, Homography as FrobNormErrHomLosserHomography
import .SKSLosserModule: computeLoss as SKSLosserComputeLoss, Homography as SKSLosserHomography
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


println("H_true:")
println(H_true.H)
println("H_computed:")
println(H_computed.H)
println("Frob Loss: $(computeLoss(H_true, H_computed))")
println("Linf Loss: $(LINFLosserComputeLoss(LINFLosserHomography(H_true.H), LINFLosserHomography(H_computed.H)))")
println("Repr Loss: $(ReprErrLossrComputeLoss(ReprErrLossrHomography(H_true.H), ReprErrLossrHomography(H_computed.H), size(img, 2), size(img, 1)))")
println("Frob Norm Error Homography Loss: $(FrobNormErrHomLosserComputeLoss(FrobNormErrHomLosserHomography(H_true.H), FrobNormErrHomLosserHomography(H_computed.H)))")
println("SKS Loss: $(SKSLosserComputeLoss(SKSLosserHomography(H_true.H), SKSLosserHomography(H_computed.H), size(img, 2), size(img, 1)))")

# Warp the image
warped_img_true = ImageWarperModule.warpImage(img, ImageWarperModule.Homography(H_true.H))
warped_img_computed = ImageWarperModule.warpImage(img, ImageWarperModule.Homography(H_computed.H))

# Plot the images as plots with titles
p1 = heatmap(img, title="Original", axis=false)
p2 = heatmap(warped_img_true, title="Warped True", axis=false)
p3 = heatmap(warped_img_computed, title="Warped Computed", axis=false)

img_width, img_height = size(img)
plot(p1, p2, p3, layout=(3, 1), size=(img_width, img_height * 3), fmt=:png)
