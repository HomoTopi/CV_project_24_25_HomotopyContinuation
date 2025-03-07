include("rectifier.jl")

module HomotopyContinuationRectifier
    using ..HomographyModule
    using ..ConicsModule
    import Main.RectifierModule: rectify

    # Extend the function from Main.RectifierModule using its absolute qualified name
    function Main.RectifierModule.rectify(C_img::Conics)::Homography
        println("Rectifying using Homotopy Continuation method")
        return Homography(rand(3,3))
    end
end