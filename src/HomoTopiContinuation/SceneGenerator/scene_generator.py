import numpy as np
from HomoTopiContinuation.DataStructures.datastructures import SceneDescription, Conic, Conics, Circle, Homography, Img
from HomoTopiContinuation.Plotter.plotter import Plotter

class SceneGenerator:
    """
    Class for generating scenes.

    This class creates scenes from the given parameters.
    """

    def generate_scene(self, scene_description: SceneDescription) -> Img:
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
        plotter = Plotter()
        # Plot the true conics
        plotter.plot_conics(Conics(C1_true, C2_true),"True Conics")
        # Compute the homography
        H = self.compute_H(scene_description)
        H_inv = H.inv()
        # Apply homography to the true conics
        C1 = Conic(H_inv.T @ C1_true.M @ H_inv)
        C2 = Conic(H_inv.T @ C2_true.M @ H_inv)
        conics = Conics(C1, C2)
        # Plot the transformed conics
        plotter.plot_conics(conics,"Transformed Conics")
        return Img(H,conics)

    
    def compute_H(self, scene_description: SceneDescription) -> Homography:
        """
        Compute the homography matrix from the scene description using plane-to-image homography.

        Args:
            scene_description (SceneDescription): The description of the scene

        Returns:
            Homography: The homography
        """
        # Focal length
        f = scene_description.f
        # Convert theta to radians
        theta = np.radians(scene_description.theta)

        # Intrinsic matrix (assuming natural camera and principal point at (0,0))
        K = np.array([[f, 0, 0],
                    [0, f, 0],
                    [0, 0, 1]])
        
        # Reference frame
        r_pi1 = np.array([1, 0, 0])
        r_p12 = np.array([0, np.cos(theta), np.sin(theta)])
        o_pi = np.array([0, 0, 1])

        # Reference matrix
        referenceMatrix = np.array([r_pi1, r_p12, o_pi]).T

        return Homography(K @ referenceMatrix)



