import numpy as np
from HomoTopiContinuation.DataStructures.datastructures import SceneDescription, Conic, Conics, Circle, Homography, Img, DistortionParams
import HomoTopiContinuation.Plotter.Plotter as Plotter
import json
import logging
import matplotlib.pyplot as plt
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

    def generate_scene(self, scene_description: SceneDescription, nPoints: int = 100, distortion_Params = None, debug: bool = False) -> Img:
        """
        Construct the matrices of the two circles and the homography from the scene description.
        Apply the homography to the true conics (circles) to get the warped conics.
        """
        
    
        # Convert circles to conics
        C1_true = scene_description.circle1.to_conic()
        C2_true = scene_description.circle2.to_conic()
        C3_true = scene_description.circle3.to_conic()

        # Compute the homography
        H = SceneGenerator.compute_H(scene_description)

        
        
        if distortion_Params is not None:
            self.logger.info("Distortion params provided, using distorted conics")
            print("Distortion params provided, using distorted conics")
            # sample some points on the circles to be used for the distortion
            C1_points = scene_description.circle1.sample_points(100)
            C2_points = scene_description.circle2.sample_points(100)
            C3_points = scene_description.circle3.sample_points(100)
            
            if debug:
                # plot the points with matplotlib
                plt.scatter(C1_points[:, 0], C1_points[:, 1], c='red', label='C1')
                plt.scatter(C2_points[:, 0], C2_points[:, 1], c='blue', label='C2')
                plt.scatter(C3_points[:, 0], C3_points[:, 1], c='green', label='C3')
                plt.axis('equal')
                plt.legend()
                plt.show()
            
            # fit the homography
            C1_points = np.dot(H.H, C1_points.T).T
            C2_points = np.dot(H.H, C2_points.T).T
            C3_points = np.dot(H.H, C3_points.T).T
            
        
            C1_points = C1_points / C1_points[:, [2]]
            C2_points = C2_points / C2_points[:, [2]]
            C3_points = C3_points / C3_points[:, [2]]
            
            # apply distortion to the points
            C1_points = self._apply_distortion(C1_points, distortion_Params, scene_description)
            C2_points = self._apply_distortion(C2_points, distortion_Params, scene_description)
            C3_points = self._apply_distortion(C3_points, distortion_Params, scene_description)
            
            if debug:
                # plot the points with matplotlib
                plt.scatter(C1_points[:, 0], C1_points[:, 1], c='red', label='C1')
                plt.scatter(C2_points[:, 0], C2_points[:, 1], c='blue', label='C2')
                plt.scatter(C3_points[:, 0], C3_points[:, 1], c='green', label='C3')
                plt.axis('equal')
                plt.legend()
                plt.show()
            
            
            
            # fit the Conics from the distorted points
            C1 = Conic.fit_conic(C1_points)
            C2 = Conic.fit_conic(C2_points)
            C3 = Conic.fit_conic(C3_points)
            
            # randomize the conics
            #C1 = C1.randomize(scene_description.noiseScale)
            #C2 = C2.randomize(scene_description.noiseScale)
            #C3 = C3.randomize(scene_description.noiseScale)
        else:
            self.logger.info("No distortion params provided, using true conics")
            print("No distortion params provided, using true conics")
            C1 = C1_true.applyHomography(H).randomize(scene_description.noiseScale)
            C2 = C2_true.applyHomography(H).randomize(scene_description.noiseScale)
            C3 = C3_true.applyHomography(H).randomize(scene_description.noiseScale)

        conics = Conics(C1, C2, C3)

        # Randomize the conics
        C1_random = C1.randomize(scene_description.noiseScale)
        C2_random = C2.randomize(scene_description.noiseScale)
        C3_random = C3.randomize(scene_description.noiseScale)

        conics_random = Conics(C1_random, C2_random, C3_random)

        # Create the Img object and store its JSON
        return Img(H, conics, conics_random)

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
        # Convert rotations from degrees to radians
        x_rotation = np.radians(scene_description.x_rotation)
        y_rotation = np.radians(scene_description.y_rotation)

        # Intrinsic matrix (assuming natural camera and principal point at (0,0))
        K = np.array([[f, 0, 0],
                      [0, f, 0],
                      [0, 0, 1]])

        # Reference frame
        # First compute the rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(x_rotation), -np.sin(x_rotation)],
                      [0, np.sin(x_rotation), np.cos(x_rotation)]])
        
        Ry = np.array([[np.cos(y_rotation), 0, np.sin(y_rotation)],
                      [0, 1, 0],
                      [-np.sin(y_rotation), 0, np.cos(y_rotation)]])

        # Combined rotation (first x, then y)
        R = Ry @ Rx

        # Get the reference frame vectors from the rotation matrix
        r_pi1 = R[:, 0]  # First column of rotation matrix (x-axis)
        r_pi2 = R[:, 1]  # Second column of rotation matrix (y-axis)
        o_pi = scene_description.offset

        # Reference matrix
        referenceMatrix = np.array([r_pi1, r_pi2, o_pi]).T

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
    
    def _apply_distortion(self, pts: np.ndarray, distortion_Params: DistortionParams, scene_description: SceneDescription) -> np.ndarray:
        """
        Apply the distortion to the points.
        """
        
        fx = scene_description.f
        fy = scene_description.f
        cx = 0
        cy = 0
        
        
        k1, k2, p1, p2, k3 = distortion_Params.k1, distortion_Params.k2, distortion_Params.p1, distortion_Params.p2, distortion_Params.k3

        # Convert pixels to normalized camera coords
        x = (pts[:, 0] - cx) / fx
        y = (pts[:, 1] - cy) / fy

        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2 * r4

        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        x_radial = x * radial
        y_radial = y * radial

        # Tangential distortion
        x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        # Total distorted normalized coords
        x_dist = x_radial + x_tangential
        y_dist = y_radial + y_tangential

        # Back to pixel coordinates
        u_dist = fx * x_dist + cx
        v_dist = fy * y_dist + cy

        return np.stack([u_dist, v_dist, np.ones(len(u_dist))], axis=-1)

if (__name__ == "__main__"):
    scene_generator = SceneGenerator()
    # Assuming scene_description is already defined
    # scene_description = SceneDescription(...)  # Replace with actual scene description
    # Call the generate_scene method
    # scene_generator.generate_scene(scene_description)
    # For demonstration, we will use dummy data
    C1 = Circle(np.array([0, 0]), 1)
    C2 = Circle(np.array([1, 1]), 1)
    C3 = Circle(np.array([2, 2]), 1)
    scene_description = SceneDescription(1, 45, np.array([0, 0]), C1, C2, C3)
    scene_generator.generate_scene(scene_description)
