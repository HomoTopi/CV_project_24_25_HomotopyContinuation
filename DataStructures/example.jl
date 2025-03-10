

module mainExample
    include("Datastructures.jl")

    import .Datastructures: SceneDescription, Conics, Conic, Homography, Image

    h = Homography(rand(3,3))
    cc = Conic(rand(3,3))
    c = Conics(cc, cc)
    i = Image(h, c)
    s = SceneDescription(1,2)
    display(s)
    display(i)
    display(h)
    display(c)
end