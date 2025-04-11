import numpy as np
from HomoTopiContinuation.DataStructures.datastructures import SceneDescription, Conic, Conics, Circle, Homography, Img
import HomoTopiContinuation.Plotter.Plotter as Plotter
import json
import logging


class SceneGenerator:
    """
    Class for generating scenes.

    This class creates scenes from the given parameters.
    It generates conics (circles) and computes the homography matrix
    based on the scene description.
    
    """
    def __init__(self):
        """
        Initialize the SceneGenerator class.
        Set up the logger for the class.
        """
        logging.basicConfig(
            filename='sceneGenerator.log',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
            )
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def generate_scene():
        """
        Construct the matrices of the two circles and the homography from the scene description.
        Apply the homography to the true conics (circles) to get the warped conics.
        """
        logger = logging.getLogger("SceneGenerator")

        # Load all scene descriptions from file
        with open('src\\HomoTopiContinuation\\Data\\sceneDescription.json', 'r') as file:
            data = json.load(file)  # list of sceneDescription JSON objects

        all_images = []

        for scene_json in data:
            scene_description = SceneDescription.from_json(scene_json)

            # Convert circles to conics
            C1_true = scene_description.circle1.to_conic()
            C2_true = scene_description.circle2.to_conic()
            C3_true = scene_description.circle3.to_conic()

            logger.info(f"Circle 1: {C1_true.M}")
            logger.info(f"Circle 2: {C2_true.M}")   
            logger.info(f"Circle 3: {C3_true.M}")

            # Compute the homography
            H = SceneGenerator.compute_H(scene_description)

            logger.info(f"Homography matrix: {H.H}")

            # Apply homography to the true conics
            C1 = C1_true.applyHomography(H)
            C2 = C2_true.applyHomography(H)
            C3 = C3_true.applyHomography(H)

            logger.info(f"Warped conic 1: {C1.M}")
            logger.info(f"Warped conic 2: {C2.M}")
            logger.info(f"Warped conic 3: {C3.M}")

            conics = Conics(C1, C2, C3)

            # Create the Img object and store its JSON
            img = Img(H, conics)
            all_images.append(img.to_json())

        # Write all images to a single JSON file
        with open('src\\HomoTopiContinuation\\Data\\sceneImage.json', 'w') as file:
            json.dump(all_images, file, indent=4)




    @staticmethod
    def compute_H(scene_description: SceneDescription) -> Homography:
        """
        Compute the homography matrix from the scene description using plane-to-image homography.
        The homography is computed multiplying the intrinsic calibration matrix and the reference matrix.
        The calibration matrix is assumed to be for a natural camera with the principal point at (0,0).
        The reference matrix represents in the world frame the x-axis, the y-axis with its rotation and the offset of the camera with respect to the plane.

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
        The reference matrix represents in the world frame the x-axis, the y-axis with its rotation and the offset of the camera with respect to the plane.

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

if(__name__ == "__main__"):
    scene_generator = SceneGenerator()
    scene_generator.generate_scene()
    