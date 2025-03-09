

module StandardRectifier
    include("Rectifier.jl")
    import .RectifierModule: rectify, Conics, Homography

    # Extend the function from Main.RectifierModule using its absolute qualified name
    function rectify(C_img::Conics)::Homography
        println("Rectifying using standard method")
        return Homography(rand(3,3))
    end
end