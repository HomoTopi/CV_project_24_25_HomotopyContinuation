

module mainExample
    include("Datastructures.jl")

    import .SceneDescriptionModule: SceneDescription
    import .HomographyModule: Homography
    import .ConicsModule: Conics
    import .ImageModule: Image
    h = Homography(rand(3,3))
    c = Conics(rand(3,3), rand(3,3))
    i = Image(h, c)
    s = SceneDescription(1,2)
    display(s)
    display(i)
    display(h)
    display(c)
end