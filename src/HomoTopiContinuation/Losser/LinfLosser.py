from HomoTopiContinuation.Losser.Losser import Losser
from HomoTopiContinuation.DataStructures.datastructures import Homography
import numpy as np


class LinfLosser(Losser):
    """
    Class to compute the loss between two homographies using the angle between the lines at infinity.
    """

    def computeLoss(h_true: Homography, h_computed: Homography) -> float:
        """
        Compute the loss between two homographies using the angle between the lines at infinity.

        Args:
            h_true (Homography): The true homography
            h_computed (Homography): The computed homography

        Returns:
            float: The loss between the two homographies
        """
        l_inf_true = h_true()[2, :]
        l_inf_true_normal = l_inf_true[:2]

        l_inf_computed = h_computed()[2, :]
        l_inf_computed_normal = l_inf_computed[:2]

        angle = None
        if np.linalg.norm(l_inf_true_normal) > 0 and np.linalg.norm(l_inf_computed_normal
                                                                    ) > 0:
            angle = np.arccos(np.abs(l_inf_true_normal @ l_inf_computed_normal) / (np.linalg.norm(
                l_inf_true_normal) * np.linalg.norm(l_inf_computed_normal))) * 180 / np.pi

        return angle
