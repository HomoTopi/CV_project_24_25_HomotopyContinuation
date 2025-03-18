from HomoTopiContinuation.Losser.Losser import Losser
from HomoTopiContinuation.DataStructures.datastructures import Homography
import numpy as np


class ReconstructionErrorLosser(Losser):
    """
    Class to compute the loss between two homographies using the Reconstruction Error.
    """

    def computeLoss(h_true: Homography, h_computed: Homography, points: np.ndarray) -> float:
        warped_points_true = h_true().dot(points)
        warped_points_computed = h_computed().dot(points)
        return np.linalg.norm(warped_points_true - warped_points_computed)
