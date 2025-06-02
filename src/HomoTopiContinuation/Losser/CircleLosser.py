from HomoTopiContinuation.Losser.Losser import SceneLosser
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic, SceneDescription
import numpy as np
import logging


class CircleLosser(SceneLosser):
    """
    Class to compute the loss based on the reconstructed circles.
    """

    def computeEccentricity(conic: Conic) -> float:
        """
        Computes the eccentricity of a conic.

        Args:
            conic (Conic): The conic to compute the eccentricity for

        Returns:
            float: The eccentricity of the conic
        """
        # Check if the conic is degenerate
        if np.linalg.det(conic.M) == 0:
            return float('inf')

        A = conic.M[:2, :2]

        eigs, _ = np.linalg.eig(A)
        det = np.linalg.det(A)
        denoms = 1/eigs
        if det > 0:
            # ellipse
            return np.sqrt(1 - np.min(denoms)/np.max(denoms))

        # Hyperbola
        return np.sqrt(1 + np.abs(np.min(denoms)/np.max(denoms)))

    def computeCircleLoss(sceneDescription: SceneDescription, C_computed: Conics) -> float:
        """
        Computes the loss of a shape reconstruction based on the reconstracted circles.
        This returns the eccentricities of the circles e.
        When the circles are perfect, e = 0.
        When the circles are ellipses, 0 < e < 1.
        When the circles are hyperbolas, e > 1.
        When the circles are parabolas, e = 1.
        When the circles are degenerate, e = ∞.
        The closer e is to 0, the better the reconstruction.

        Args:
            sceneDescription (SceneDescription): The scene description
            C_computed (Conics): The reconstructed Circles

        Returns:
            float: The loss of the shape reconstruction
        """
        # Compute the eccentricities of the circles
        eccentricities = [
            CircleLosser.computeEccentricity(c) for c in C_computed]
        return eccentricities
