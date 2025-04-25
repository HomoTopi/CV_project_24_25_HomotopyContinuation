from HomoTopiContinuation.DataStructures.datastructures import Circle
import numpy as np
import HomoTopiContinuation.Plotter.Plotter as Plotter
import HomoTopiContinuation.SceneGenerator.scene_generator as sg
import HomoTopiContinuation.Rectifier.standard_rectifier as sr
import HomoTopiContinuation.Rectifier.homotopyc_rectifier as hr
import HomoTopiContinuation.Rectifier.numeric_rectifier as nr
from HomoTopiContinuation.Losser.CircleLosser import CircleLosser
from HomoTopiContinuation.ConicWarper.ConicWarper import ConicWarper
from enum import Enum


class Rectifiers(Enum):
    standard = sr.StandardRectifier()
    homotopy = hr.HomotopyContinuationRectifier()
    numeric = nr.NumericRectifier()


def sceneDefinition() -> sg.SceneDescription:
    # Parameters
    f = 8.63105337413579
    theta = 20.676281977525

    # Define the circles
    c1 = Circle(
        np.array([3.63394461831874, 5.87477335920174]), 2.4463646596343)
    c2 = Circle(
        np.array([9.12957268926517, 3.89972677623066]), 3.32330513800954)
    c3 = Circle(
        np.array([1.22469539983364, 5.54286132063891]), 6.16125996349048)

    print("Circle 1:")
    print(c1.to_conic().M)
    print("Circle 2:")
    print(c2.to_conic().M)
    print("Circle 3:")
    print(c3.to_conic().M)

    offset = np.array([0, 0, 2])
    noiseScale = 0

    return sg.SceneDescription(f, theta, offset, c1, c2, c3, noiseScale)


def main():
    rectifier = Rectifiers.numeric.value
    losser = CircleLosser

    sceneDescription = sceneDefinition()
    print("[Scene Described]")

    img = sg.SceneGenerator().generate_scene(sceneDescription)
    print("[Scene Generated]")

    try:
        H_reconstructed = rectifier.rectify(img.C_img)
    except Exception as e:
        print("[Rectification Failed]")
        print("Error:")
        print(e)
        # Plot the results
        plotter = Plotter.Plotter(2, 2, title="Experiment")

        plotter.plotScene(sceneDescription, img)

        plotter.show()
        return

    print("True Homography:")
    print(img.h_true.H)

    print("Reconstructed Homography:")
    print(H_reconstructed.H)

    # Warp The Circles
    warpedConics = ConicWarper().warpConics(img.C_img, H_reconstructed)
    print("[Conics Warped]")
    print("Warped Conics:")
    print(warpedConics.C1.M)
    print(warpedConics.C2.M)
    print(warpedConics.C3.M)

    # Compute the loss
    loss = losser.computeCircleLoss(sceneDescription, warpedConics)
    print("Loss:")
    print(loss)

    # Plot the results
    plotter = Plotter.Plotter(2, 2, title="Experiment")

    plotter.plotExperiment(sceneDescription, img,
                           warpedConics)

    plotter.show()


if __name__ == "__main__":
    main()
