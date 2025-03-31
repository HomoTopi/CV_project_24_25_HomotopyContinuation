from HomoTopiContinuation.DataStructures.datastructures import Circle
import numpy as np
import HomoTopiContinuation.Plotter.Plotter as Plotter
import HomoTopiContinuation.SceneGenerator.scene_generator as sg
import HomoTopiContinuation.Rectifier.standard_rectifier as sr


def sceneDefinition() -> sg.SceneDescription:
    # Parameters
    f = 1
    theta = 45

    # Define the circles
    c1 = Circle(np.array([0, 0]), 1)
    c2 = Circle(np.array([0.5, 0]), 1)
    c3 = Circle(np.array([0, 0]), 1.5)
    offset = np.array([0, 0, 2])

    return sg.SceneDescription(f, theta, offset, c1, c2, c3)


def main():
    sceneDescription = sceneDefinition()
    print("[Scene Described]")

    img = sg.SceneGenerator.generate_scene(sceneDescription)
    print("[Scene Generated]")

    rectifier = sr.StandardRectifier()

    H_reconstructed = rectifier.rectify(img.C_img)
    print("[Rectified]")

    print("True Homography:")
    print(img.h_true.H)

    print("Reconstructed Homography:")
    print(H_reconstructed.H)

    # Warp The Circles
    C1_reconstructed = img.C_img.C1.applyHomography(H_reconstructed)
    C2_reconstructed = img.C_img.C2.applyHomography(H_reconstructed)
    C3_reconstructed = img.C_img.C3.applyHomography(H_reconstructed)

    plotter = Plotter.Plotter(3, 1, title="Experiment")

    plotter.plotScene(sceneDescription, img)
    min_x, max_x = -3, 3
    min_y, max_y = -3, 3
    plotter.plotConic2D(
        C1_reconstructed, conicName="", color="red", x_range=(min_x, max_x, 500), y_range=(min_y, max_y, 500))
    plotter.plotConic2D(
        C2_reconstructed, conicName="", color="green", x_range=(min_x, max_x, 500), y_range=(min_y, max_y, 500))
    plotter.plotConic2D(
        C3_reconstructed, conicName="", color="blue", x_range=(min_x, max_x, 500), y_range=(min_y, max_y, 500))

    plotter.show()


if __name__ == "__main__":
    main()
