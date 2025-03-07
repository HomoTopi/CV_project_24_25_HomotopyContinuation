include("standardRectifier.jl")

C_img = Main.ConicsModule.Conics(rand(3,3), rand(3,3))
try
    h = Main.RectifierModule.rectify(C_img)
    display(h.H)
catch e
    println("Error: $e")
end
