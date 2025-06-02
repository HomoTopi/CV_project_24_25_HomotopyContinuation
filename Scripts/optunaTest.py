import optuna
import os
from dotenv import load_dotenv
import numpy as np
from HomoTopiContinuation.Losser import CircleLosser
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic, SceneDescription, Circle
from HomoTopiContinuation.SceneGenerator.scene_generator import SceneGenerator
from HomoTopiContinuation.Rectifier.homotopyc_rectifier import HomotopyContinuationRectifier
from HomoTopiContinuation.Rectifier.standard_rectifier import StandardRectifier
from HomoTopiContinuation.Rectifier.numeric_rectifier import NumericRectifier
from HomoTopiContinuation.ConicWarper.ConicWarper import ConicWarper


def objective(trial):
    """
    Objective function for Optuna to optimize the parameters of the scene generation.
    The function generates a scene with the given parameters, rectifies the image and warps the conics in the scene,
    and computes the eccentricities of the warped conics.
    The function returns the negative sum of the eccentricities as the objective value to minimize.
    """
    maxPos = 500
    maxRadius = 100
    f = trial.suggest_float("f", 1, 5)
    y_rotation = trial.suggest_float("y_rotation", 5, 20)
    offset_x = trial.suggest_float("offset_x", 0, 0)
    offset_y = trial.suggest_float("offset_y", 0, 0)
    offset_z = trial.suggest_float("offset_z", 2, 2)
    c1_centre_x = trial.suggest_float("c1_centre_x", 0, maxPos)
    c1_centre_y = trial.suggest_float("c1_centre_y", 0, maxPos)
    c1_radius = trial.suggest_float("c1_radius", 1, maxRadius)
    c2_centre_x = trial.suggest_float("c2_centre_x", 0, maxPos)
    c2_centre_y = trial.suggest_float("c2_centre_y", 0, maxPos)
    c2_radius = trial.suggest_float("c2_radius", 1, maxRadius)
    c3_centre_x = trial.suggest_float("c3_centre_x", 0, maxPos)
    c3_centre_y = trial.suggest_float("c3_centre_y", 0, maxPos)
    c3_radius = trial.suggest_float("c3_radius", 1, maxRadius)
    
    
    ## distortion params ###
    
    # Radial distortion
    k1 = trial.suggest_uniform('k1', -1.5, 0.0)    # strong barrel → mild
    k2 = trial.suggest_uniform('k2', -0.5, 0.5)    # second‑order radial
    k3 = trial.suggest_uniform('k3', -0.1, 0.1)    # optional 6th‑order term

    # Tangential distortion
    p1 = trial.suggest_uniform('p1', -0.01, 0.01)  # usually very small
    p2 = trial.suggest_uniform('p2', -0.01, 0.01)

    try:
        circle1 = Circle(np.array([c1_centre_x, c1_centre_y]), c1_radius)
        circle2 = Circle(np.array([c2_centre_x, c2_centre_y]), c2_radius)
        circle3 = Circle(np.array([c3_centre_x, c3_centre_y]), c3_radius)
        scene = SceneDescription(f, y_rotation, np.array([offset_x, offset_y, offset_z]),
                                 circle1, circle2, circle3)
        scene_generator = SceneGenerator()
        image = scene_generator.generate_scene(scene)
        rectifier = HomotopyContinuationRectifier()
        homography = rectifier.rectify(image.C_img)
        conic_warper = ConicWarper()
        warpedConics = conic_warper.warpConics(image.C_img, homography)
        eccentricities = CircleLosser.CircleLosser.computeCircleLoss(
            scene, warpedConics)
        return max(eccentricities)
    except ValueError as e:
        # TODO: How to handle the exceptions appropriately?
        raise optuna.exceptions.TrialPruned()


def experiment():
    """
    Run the experiments with Optuna and store the results on supabase db.
    """
    load_dotenv()

    storage = (os.getenv("OPTUNA_DB_URL"))
    search_space = {"f": [1, 10.0], "y_rotation": [0, 50],  "offset_x": [0, 0], "offset_y": [0, 0],
                    "offset_z": [1.0, 1.0], "c1_centre_x": [0, 50.0], "c1_centre_y": [0.0, 50],
                    "c1_radius": [1.0, 50], "c2_centre_x": [0.0, 50], "c2_centre_y": [0.0, 50],
                    "c2_radius": [1.0, 50], "c3_centre_x": [0.0, 50], "c3_centre_y": [0.0, 50],
                    "c3_radius": [1.0, 50]}
    seed = 42
    study = optuna.create_study(
        study_name="scene_opt_maximize_randomSampler_HomotopicRectifier_Not centered",
        storage=storage,
        load_if_exists=True,
        direction="maximize"
        # sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    study.optimize(objective, n_trials=300)


if __name__ == "__main__":
    experiment()
