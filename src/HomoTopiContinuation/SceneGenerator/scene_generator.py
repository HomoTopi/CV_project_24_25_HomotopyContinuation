import numpy as np
from HomoTopiContinuation.DataStructures.datastructures import SceneDescription, Conic, Conics, Circle, Homography, Img
from HomoTopiContinuation.Plotter.plotter import Plotter


class SceneGenerator:
    """
    Class for generating scenes.

    This class creates scenes from the given parameters.
    """
    @staticmethod
    def generate_scene(scene_description: SceneDescription) -> Img:
        """
        Generate a scene from the given description.

        Args:
            scene_description (SceneDescription): The description of the scene

        Returns:
            Img: The pair of conics and the true homography
        """
        # Convert circles to conics
        C1_true = scene_description.circle1.to_conic()
        C2_true = scene_description.circle2.to_conic()
        # Compute the homography
        H = SceneGenerator.compute_H(scene_description)
        H_inv = H.inv()
        # Apply homography to the true conics
        C1 = Conic(H_inv.T @ C1_true.M @ H_inv)
        C2 = Conic(H_inv.T @ C2_true.M @ H_inv)
        conics = Conics(C1, C2)
        return Img(H, conics)

    @staticmethod
    def compute_H(scene_description: SceneDescription) -> Homography:
        """
        Compute the homography matrix from the scene description using plane-to-image homography.

        Args:
            scene_description (SceneDescription): The description of the scene

        Returns:
            Homography: The homography
        """
        # Focal length
        f = scene_description.f
        # Convert y_rotation from degrees to radians
        y_rotation = np.radians(scene_description.y_rotation)

        # Intrinsic matrix (assuming natural camera and principal point at (0,0))
        K = np.array([[f, 0, 0],
                      [0, f, 0],
                      [0, 0, 1]])

        # Reference frame
        r_pi1 = np.array([1, 0, 0])
        r_p12 = np.array([0, np.cos(y_rotation), np.sin(y_rotation)])
        o_pi = scene_description.offset

        # Reference matrix
        referenceMatrix = np.array([r_pi1, r_p12, o_pi]).T

        return Homography(K @ referenceMatrix)

    @staticmethod
    def compute_reference_matrix(scene_description: SceneDescription) -> np.ndarray:
        """
        Compute the reference matrix from the scene description.

        Args:
            scene_description (SceneDescription): The description of the scene

        Returns:
            np.ndarray: The reference matrix
        """
        # Convert y_rotation from degrees to radians
        y_rotation = np.radians(scene_description.y_rotation)

        # Reference frame
        r_pi1 = np.array([1, 0, 0])
        r_p12 = np.array([0, np.cos(y_rotation), np.sin(y_rotation)])
        o_pi = scene_description.offset

        return np.array([r_pi1, r_p12, o_pi]).T
