from HomoTopiContinuation.DataStructures.datastructures import Circle
import numpy as np
import HomoTopiContinuation.Plotter.Plotter as Plotter
import HomoTopiContinuation.SceneGenerator.scene_generator as sg
import HomoTopiContinuation.Rectifier.standard_rectifier as sr
import HomoTopiContinuation.Rectifier.homotopyc_rectifier as hr
import HomoTopiContinuation.Rectifier.numeric_rectifier as nr
from HomoTopiContinuation.Losser.CircleLosser import CircleLosser
from HomoTopiContinuation.ConicWarper.ConicWarper import ConicWarper
from HomoTopiContinuation.Losser.CPLosser import CPLosser
from enum import Enum


class Rectifiers(Enum):
    standard = sr.StandardRectifier()
    homotopy = hr.HomotopyContinuationRectifier()
    numeric = nr.NumericRectifier()


def sceneDefinition() -> sg.SceneDescription:
    # Parameters
    f = 1
    theta = 30

    # Define the circles
    c1 = Circle(
        np.array([-20, 0]), 5)
    c2 = Circle(
        np.array([0, 2]), 5)
    c3 = Circle(
        np.array([20, -4]), 5)

    print("Circle 1:")
    print(c1.to_conic().M)
    print([float(p) for p in c1.to_conic().to_algebraic_form()])
    print("Circle 2:")
    print(c2.to_conic().M)
    print([float(p) for p in c2.to_conic().to_algebraic_form()])
    print("Circle 3:")
    print(c3.to_conic().M)
    print([float(p) for p in c3.to_conic().to_algebraic_form()])

    offset = np.array([0, 0, 10])
    noiseScale = 0.00

    return sg.SceneDescription(f, theta, offset, c1, c2, c3, noiseScale)


def main():
    rectifier = Rectifiers.homotopy.value
    losser = CircleLosser

    sceneDescription = sceneDefinition()
    print("[Scene Described]")

    img = sg.SceneGenerator().generate_scene(sceneDescription)
    print("[Scene Generated]")

    print("[Image of the Circular points]:", img.imCircularPoints)

    try:
        H_reconstructed, imCPsReconstructed = rectifier.rectify(
            img.C_img_noise, returnCP=True)
        print("[Rectification Successful]")
        print("Reconstructed Image of the Circular points:")
        print(imCPsReconstructed)
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
    print(img.h_true.H / img.h_true.H[2, 2])

    print("Reconstructed Homography:")
    print(H_reconstructed.H / H_reconstructed.H[2, 2])

    # Warp The Circles
    warpedConics = ConicWarper().warpConics(img.C_img, H_reconstructed)
    print("[Conics Warped]")
    print("Warped Conics:")
    print(warpedConics.C1.M)
    print(warpedConics.C2.M)
    print(warpedConics.C3.M)

    # Compute the loss
    originalLoss = losser.computeCircleLoss(sceneDescription, img.C_img)
    print("Original Loss:")
    print(originalLoss)

    loss = losser.computeCircleLoss(sceneDescription, warpedConics)
    print("Loss:")
    print(loss)

    cpLoss = CPLosser.computeLoss(img.imCircularPoints.T, imCPsReconstructed)
    print("Loss of the Circular Points:")
    print(cpLoss)

    print("Repositioned Circular Points:")
    ripositionedCP = H_reconstructed.H @ imCPsReconstructed.T
    ripositionedCP = ripositionedCP.T / ripositionedCP.T[:, [0]]
    print(ripositionedCP)

    # Plot the results
    plotter = Plotter.Plotter(2, 2, title="Experiment")

    plotter.plotExperiment(sceneDescription, img,
                           warpedConics)

    plotter.show()


if __name__ == "__main__":
    main()
