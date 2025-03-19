from HomoTopiContinuation.Losser.Losser import Losser
from HomoTopiContinuation.DataStructures.datastructures import Homography
import numpy as np


class AngleDistortionLosser(Losser):
    """
    Class to compute the loss between two homographies using the cosine of the angle between two perpendicular lines.
    """

    C_inf_star = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0.0]
    ])

    def distanceMetric(C: np.ndarray, line_1: np.ndarray, line_2: np.ndarray) -> float:
        return line_1.T @ C @ line_2

    def computeLoss(h_true: Homography, h_computed: Homography) -> float:
        """
        Compute the loss between two homographies using the cosine of the angle between two perpendicular lines.

        Args:
            h_true (Homography): The true homography
            h_computed (Homography): The computed homography

        Returns:
            float: The loss between the two homographies
        """
        l = np.array([1, 0, 0])
        m = np.array([0, 1, 0])
        C = h_true() @ h_computed.inv() @ \
            AngleDistortionLosser.C_inf_star @ h_computed.inv().T @ h_true().T
        print('\tC_inf_star\n', C)
        return np.acos(AngleDistortionLosser.distanceMetric(C, l, m) / np.sqrt(AngleDistortionLosser.distanceMetric(C, l, l) * AngleDistortionLosser.distanceMetric(C, m, m))) * 180 / np.pi
