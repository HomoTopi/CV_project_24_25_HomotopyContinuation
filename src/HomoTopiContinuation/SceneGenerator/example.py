from HomoTopiContinuation.DataStructures.datastructures import SceneDescription, Circle, Img
from HomoTopiContinuation.SceneGenerator.scene_generator import SceneGenerator
from HomoTopiContinuation.Plotter.plotter import Plotter
import numpy as np

def main():
    """
    Example demonstrating the usage of the scene generator
    """
    # Create two circles
    # These are just example circles - replace with real data for actual use
    center1 = np.array([0.0, 0.0])
    radius1 = 1.0
    circle1 = Circle(center1, radius1)

    center2 = np.array([0.0, 0.0])
    radius2 = 1.5
    circle2 = Circle(center2, radius2)

    # Create a SceneDescription object
    f = 1  # Focal length
    theta = 30.0 # Rotation angle
    scene_description = SceneDescription(f,theta,circle1,circle2)

    # Create the scene generator
    scene_generator = SceneGenerator()

    # Generate the scene
    img = scene_generator.generate_scene(scene_description)

    print("Scene generated successfully!")
if __name__ == "__main__":
    main()