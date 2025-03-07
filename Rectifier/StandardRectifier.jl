include("rectifier.jl")

module StandardRectifier
    using ..HomographyModule
    using ..ConicsModule
    import Main.RectifierModule: rectify

    # Extend the function from Main.RectifierModule using its absolute qualified name
    function Main.RectifierModule.rectify(C_img::Conics)::Homography
        println("Rectifying using standard method")
        return Homography(rand(3,3))
    end
end