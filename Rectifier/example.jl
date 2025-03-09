include("StandardRectifier.jl")
include("HomotopyRectifier.jl")

import .StandardRectifier: rectify, Conics, Homography
import .HomotopyContinuationRectifier: rectify as rectify2, Conics as Conics2, Homography as Homography2
C_img = Conics(rand(3,3), rand(3,3))
try
    H = rectify(C_img)
    println("Homography: $H")
catch e
    println("Error: $e")
end

try
    C_img2 = Conics2(rand(3,3), rand(3,3))
    H2 = rectify2(C_img2)
    println("Homography: $H2")
catch e
    println("Error: $e")
end