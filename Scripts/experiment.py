from HomoTopiContinuation.DataStructures.datastructures import Circle, DistortionParams
import numpy as np
import HomoTopiContinuation.Plotter.Plotter as Plotter
import HomoTopiContinuation.SceneGenerator.scene_generator as sg
import HomoTopiContinuation.Rectifier.standard_rectifier as sr
import HomoTopiContinuation.Rectifier.homotopyc_rectifier as hr
import HomoTopiContinuation.Rectifier.numeric_rectifier as nr
from HomoTopiContinuation.Losser.CircleLosser import CircleLosser
from HomoTopiContinuation.ConicWarper.ConicWarper import ConicWarper
from enum import Enum
from HomoTopiContinuation.Rectifier.GDRectifier import GDRectifier

class Rectifiers(Enum):
    standard = sr.StandardRectifier()
    homotopy = hr.HomotopyContinuationRectifier()
    numeric = nr.NumericRectifier()
    gd = GDRectifier()


def sceneDefinition() -> sg.SceneDescription:
    # Parameters
    f = 2
    theta = 80

    # Define the circles
    c1 = Circle(
        np.array([0, 0]), 1)
    c2 = Circle(
        np.array([2, 1]), 0.8)
    c3 = Circle(
        np.array([1, 2]), 0.5)

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
    noiseScale = 0.0003

    return sg.SceneDescription(f, theta, offset, c1, c2, c3, noiseScale, x_rotation=0)


def main():
    rectifier = Rectifiers.homotopy.value
    losser = CircleLosser
    distortion_Params = DistortionParams(k1=-0.35, k2=0.5, p1=0.001, p2=0.001, k3=0.0)
    sceneDescription = sceneDefinition()
    print("[Scene Described]")

    #img = sg.SceneGenerator().generate_scene(sceneDescription, distortion_Params=distortion_Params, debug=True)
    img = sg.SceneGenerator().generate_scene(sceneDescription, debug=True)
    print("[Scene Generated]")

    
    try:
        #H_reconstructed = rectifier.rectify(img.C_img_noise)
        H_reconstructed, history, losses, grads, ms, vs = GDRectifier.rectify(C_img=img.C_img_noise, alpha=0.075, beta1=0.5, beta2=0.9, epsilon=1e-3)
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
    
    print("Image Conics")
    
    assert img.C_img.C1.is_ellipse() , "the Image Conic 1 is not an ellipse"
    assert img.C_img.C2.is_ellipse() , "the Image Conic 2 is not an ellipse"
    assert img.C_img.C3.is_ellipse() , "the Image Conic 3 is not an ellipse"
    
    print(img.C_img.C1.M)
    print(img.C_img.C2.M)
    print(img.C_img.C3.M)

    print("Reconstructed Homography:")
    print(H_reconstructed.H / H_reconstructed.H[2, 2])

    # Warp The Circles
    warpedConics = ConicWarper().warpConics(img.C_img, H_reconstructed)
    print("[Conics Warped]")
    print("Warped Conics:")
    print(warpedConics.C1.M)
    
    # Compute the loss
    originalLoss = losser.computeCircleLoss(sceneDescription, img.C_img)
    print("Original Loss:")
    print(originalLoss)

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
