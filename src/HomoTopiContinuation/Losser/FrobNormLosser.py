from HomoTopiContinuation.Losser.Losser import Losser
from HomoTopiContinuation.DataStructures.datastructures import Homography
import numpy as np


class FrobNormLosser(Losser):
    """
    Class to compute the loss between two homographies using the Frobenius norm of the difference.
    """

    def computeLoss(h_true: Homography, h_computed: Homography) -> float:
        """
        Compute the loss between two homographies using the Frobenius norm of the difference.

        Args:
            h_true (Homography): The true homography
            h_computed (Homography): The computed homography

        Returns:
            float: The loss between the two homographies
        """
        return np.linalg.norm(h_true() - h_computed(), 'fro')
