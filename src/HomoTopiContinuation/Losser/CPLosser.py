import numpy as np


class CPLosser():
    def computeLoss(trueCP: np.ndarray, computedCP: np.ndarray) -> float:
        """
        Compute the loss between two sets of circular points.

        Args:
            trueCP (np.ndarray): The true circular points
            computedCP (np.ndarray): The computed circular points

        Returns:
            float: The loss between the two sets of circular points
        """
        normalized_trueCP = trueCP / trueCP[:, [0]]
        normalized_computedCP = computedCP / computedCP[:, [0]]

        distances = np.linalg.norm(
            normalized_trueCP - normalized_computedCP, axis=1)

        return np.max(distances)
