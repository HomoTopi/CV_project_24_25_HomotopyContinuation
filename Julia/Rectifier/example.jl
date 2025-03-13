include("StandardRectifier.jl")
include("HomotopyRectifier.jl")

import .StandardRectifier: rectify, Conics, Homography, Conic
#import .HomotopyContinuationRectifier: rectify as rectify2, Conics as Conics2, Homography as Homography2
C1 = Conic(rand(3,3))
C2 = Conic(rand(3,3))
C_img = Conics(C1, C2)
try
    H = rectify(C_img)
    println("Homography: $H")
catch e
    println("Error: $e")
end

#try
#    C_img2 = Conics2(C1, C2)
#    H2 = rectify2(C_img2)
#    println("Homography: $H2")
#catch e
#    println("Error: $e")
#end