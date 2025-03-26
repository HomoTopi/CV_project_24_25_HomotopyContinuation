from HomoTopiContinuation.DataStructures.datastructures import Circle
import numpy as np
import HomoTopiContinuation.Plotter.Plotter as Plotter
import HomoTopiContinuation.SceneGenerator.scene_generator as sg
import HomoTopiContinuation.Rectifier.standard_rectifier as sr


def sceneDefinition() -> sg.SceneDescription:
    # Parameters
    f = 1
    theta = 0

    # Define the circles
    c1 = Circle(np.array([0, 0]), 1)
    c2 = Circle(np.array([0.5, 0]), 1)
    offset = np.array([0, 0, 2])

    return sg.SceneDescription(f, theta, offset, c1, c2)


def main():
    sceneDescription = sceneDefinition()
    img = sg.SceneGenerator.generate_scene(sceneDescription)

    rectifier = sr.StandardRectifier()

    H_reconstructed = rectifier.rectify(img.C_img)

    print(H_reconstructed.H)

    # Warp The Circles
    C1_reconstructed = img.C_img.C1.applyHomography(H_reconstructed)
    C2_reconstructed = img.C_img.C2.applyHomography(H_reconstructed)

    plotter = Plotter.Plotter(3, 1, title="Experiment")

    plotter.plotScene(sceneDescription, img)
    plotter.plotConic2D(
        C1_reconstructed, conicName="Reconstructed Circle 1", color="red")
    plotter.plotConic2D(
        C2_reconstructed, conicName="Reconstructed Circle 2", color="blue")

    plotter.show()


if __name__ == "__main__":
    main()
