import json
import numpy as np
from enum import Enum
from HomoTopiContinuation.DataStructures.datastructures import Homography, Conics, SceneDescription, Img
from HomoTopiContinuation.Losser import AngleDistortionLosser, CircleLosser, FrobNormLosser, LinfLosser, ReconstructionErrorLosser

def computeLoss():
    """
    Compute the losses for the given scene and homography.
    """
    with open('src\\HomoTopiContinuation\\Data\\sceneDescription.json', 'r') as file:
        scene_data = json.load(file)
    with open('src\\HomoTopiContinuation\\Data\\sceneImage.json', 'r') as file:
        image_data = json.load(file)
    with open('src\\HomoTopiContinuation\\Data\\rectifiedHomography.json', 'r') as file:
        homography_data = json.load(file)
    with open('src\\HomoTopiContinuation\\Data\\rectifiedConics.json', 'r') as file:
        conics_data = json.load(file)

    losses = []
    for scene_json, img_json, H_json, conic_json in zip(scene_data, image_data, homography_data, conics_data):
        scene = SceneDescription.from_json(scene_json)
        img = Img.from_json(img_json)
        H = Homography.from_json(H_json)
        conics = Conics.from_json(conic_json)

        # Compute the loss for each type
        angle_loss = AngleDistortionLosser.AngleDistortionLosser.computeLoss(img.h_true, H)
        circle_loss = CircleLosser.CircleLosser.computeCircleLoss(scene, conics)
        frob_loss = FrobNormLosser.FrobNormLosser.computeLoss(img.h_true, H)
        linf_loss = LinfLosser.LinfLosser.computeLoss(img.h_true, H)

        losses.append({
            "angle_loss": angle_loss,
            "circle_loss": circle_loss,
            "frob_loss": frob_loss,
            "linf_loss": linf_loss
        })
    with open('src\\HomoTopiContinuation\\Data\\losses.json', 'w') as file:
        json.dump(losses, file, indent=4)

if __name__ == "__main__":
    computeLoss()