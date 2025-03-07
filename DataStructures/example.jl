include("datastructures.jl")

using .SceneDescriptionModule
using .ConicsModule
using .HomographyModule
using .ImageModule


h = HomographyModule.Homography(rand(3,3))
c = ConicsModule.Conics(rand(3,3), rand(3,3))
i = ImageModule.Image(h, c)

print(i)