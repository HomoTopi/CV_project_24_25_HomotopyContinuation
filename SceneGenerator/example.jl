include("../SceneGenerator/SceneGenerator.jl")
using .SceneGenerator: SceneDescription, Homography, Img, generate_scene

# Define the scene
f = 500.0
theta = 30.0
circle1 = [100.0, 100.0, 50.0]
circle2 = [200.0, 200.0, 100.0]
# Create the scene description
sd = SceneDescription(f, theta, circle1, circle2)
# Generate the scene
img = generate_scene(sd)
# Print the homography matrix
println("Homography matrix: $(img.h_true.H)")
# Print conics
println("Conic 1: $(img.C_img.C1.M)")
println("Conic 2: $(img.C_img.C2.M)")

