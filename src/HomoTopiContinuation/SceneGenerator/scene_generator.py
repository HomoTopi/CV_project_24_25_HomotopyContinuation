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
        C1_true = scene_description.circle1
        C2_true = scene_description.circle2
        print('Original conics')
        print(C1_true.M)
        print(C2_true.M)
        plotter = Plotter()
        plotter.plot_conics(Conics(C1_true, C2_true),"True Conics")
        H = self.compute_H(scene_description)
        H_inv = H.inv()
        print('Homography inv')
        print(H_inv)
         # Apply homography to the true conics
        C1 = Conic(np.transpose(H_inv) @ C1_true.M @ H_inv)
        C2 = Conic(np.transpose(H_inv) @ C2_true.M @ H_inv)
        C1.M = C1.M / C1.M[2,2]
        C2.M = C2.M / C2.M[2,2]
        conics = Conics(C1, C2)
        print('Transformed conics')
        print(conics.C1.M)
        print(conics.C2.M)
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
        f = scene_description.f
        theta = np.radians(scene_description.theta)

        # Intrinsic matrix (assuming natural camera and principal point at (0,0))
        K = np.array([[f, 0, 0],
                    [0, f, 0],
                    [0, 0, 1]])
        
        r_pi1 = np.array([1, 0, 0])
        r_p12 = np.array([0, np.cos(theta), np.sin(theta)])
        o_pi = np.array([0, 0, 1])

        referenceMatrix = np.array([r_pi1, r_p12, o_pi]).T

        # [r_pi1, r_pi2, o_pi]
        # r_pi1, r_pi2 are the unit vectors in the plane coordinate system
        # o_pi is the origin of the plane coordinate system
        # r_pi1 = [1, 0, 0]
        # r_pi2 = [0, 1, 0]
        # o_pi = [0, 0, 1]
        # matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
        #                   [np.sin(theta), np.cos(theta), 0],
        #                 [0, 0, 1]])

        # matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
        #                    [0, 1, 0],
        #                 [-np.sin(theta), 0, np.cos(theta)]])
        # R_world = np.array([[1, 0, 0],
        #                     [0, 1, 0],
        #                     [0, 0, 1]])
        # R_camera = matrix @ R_world
         
        H = K @ referenceMatrix

        return Homography(H)



