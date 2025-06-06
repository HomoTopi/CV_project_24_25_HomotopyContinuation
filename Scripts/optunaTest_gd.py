import optuna
import os
from dotenv import load_dotenv
import numpy as np
import jax.numpy as jnp
from HomoTopiContinuation.Losser import CircleLosser
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic, SceneDescription, Circle
from HomoTopiContinuation.SceneGenerator.scene_generator import SceneGenerator
from HomoTopiContinuation.Rectifier.GDRectifier import GDRectifier
from HomoTopiContinuation.Rectifier.standard_rectifier import StandardRectifier
from HomoTopiContinuation.Rectifier.numeric_rectifier import NumericRectifier
from HomoTopiContinuation.ConicWarper.ConicWarper import ConicWarper
from HomoTopiContinuation.DataStructures.datastructures import Homography

def objective(trial):
    """
    Objective function for Optuna to optimize the parameters of the scene generation.
    The function generates a scene with the given parameters, rectifies the image and warps the conics in the scene,
    and computes the eccentricities of the warped conics.
    The function returns the negative sum of the eccentricities as the objective value to minimize.
    """
    f = 2
    
    c1_centre_x, c1_centre_y, c1_radius = 0, 0, 1
    c2_centre_x, c2_centre_y, c2_radius = 2, 1, 0.8
    c3_centre_x, c3_centre_y, c3_radius = 3, 2, 0.5
    
    offset_x, offset_y, offset_z = 0, 0, 3
    
    y_rotation = trial.suggest_float("y_rotation", 5, 45)
    x_rotation = trial.suggest_float("x_rotation", 5, 45)
    
    
    ## distortion params ###
    
    # Radial distortion
    #k1 = trial.suggest_uniform('k1', -1.5, 0.0)    # strong barrel → mild
    #k2 = trial.suggest_uniform('k2', -0.5, 0.5)    # second‑order radial
    #k3 = trial.suggest_uniform('k3', -0.1, 0.1)    # optional 6th‑order term

    # Tangential distortion
    #p1 = trial.suggest_uniform('p1', -0.01, 0.01)  # usually very small
    #p2 = trial.suggest_uniform('p2', -0.01, 0.01)

    try:
        circle1 = Circle(np.array([c1_centre_x, c1_centre_y]), c1_radius)
        circle2 = Circle(np.array([c2_centre_x, c2_centre_y]), c2_radius)
        circle3 = Circle(np.array([c3_centre_x, c3_centre_y]), c3_radius)
        scene = SceneDescription(f, y_rotation, np.array([offset_x, offset_y, offset_z]),
                                 circle1, circle2, circle3, x_rotation=x_rotation)
        scene_generator = SceneGenerator()
        image = scene_generator.generate_scene(scene)        
        rectifier = GDRectifier()
        H_computed, history, losses, grads, ms, vs = GDRectifier.rectify(C_img=image.C_img_noise,
            iterations=3000,
            alpha=1e-3, 
            beta1=.99, 
            beta2=.999, 
            weights=jnp.array([1.0, 1.0, 1.0])
        )
        homography = H_computed
        assert isinstance(homography, Homography)
        homography.normalize()
        conic_warper = ConicWarper()
        warpedConics = conic_warper.warpConics(image.C_img, homography)
        
        
        eccentricities = CircleLosser.CircleLosser.computeCircleLoss(
            None, warpedConics)
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
        study_name="GD_rectifier_y_offset_x_offset",
        storage=storage,
        load_if_exists=True,
        direction="maximize"
        # sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    study.optimize(objective, n_trials=300)


if __name__ == "__main__":
    experiment()
