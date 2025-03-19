from HomoTopiContinuation.Losser.Losser import Losser
from HomoTopiContinuation.DataStructures.datastructures import Homography
import numpy as np


class ReconstructionErrorLosser(Losser):
    """
    Class to compute the loss between two homographies using the Reconstruction Error.
    """

    def computeLoss(h_true: Homography, h_computed: Homography, points: np.ndarray) -> float:
        """
        Compute the loss between two homographies using the Reconstruction Error. It is defined as the L2 norm of the difference between the warped points using the true and the computed homographies.

        Args:
            h_true (Homography): The true homography
            h_computed (Homography): The computed homography
            points (np.ndarray): The points to be warped

        Returns:
            float: The loss between the two homographies
        """
        warped_points_true = h_true().dot(points)
        warped_points_computed = h_computed().dot(points)
        return np.linalg.norm(warped_points_true - warped_points_computed, axis=0).mean()
