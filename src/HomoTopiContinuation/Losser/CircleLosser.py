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
        # Compute the eigenvalues of the conic matrix
        eigenvalues, _ = np.linalg.eig(conic.M[:2, :2])

        # Check if the conic is degenerate
        if np.isclose(np.linalg.det(conic.M), 0):
            return float('inf')

        # Compute the eccentricity
        return np.sqrt(1 - (min(eigenvalues) / max(eigenvalues)))

    def computeCircleLoss(sceneDescription: SceneDescription, C_computed: Conics) -> float:
        """
        Computes the loss of a shape reconstruction based on the reconstracted circles.
        This returns the average eccentricity of the circles e.
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
        # Compute the average eccentricity of the circles
        eccentricities = [
            CircleLosser.computeEccentricity(c) for c in C_computed]
        # print(f"Eccentricities: {eccentricities}")
        avgEccentricity = np.mean(eccentricities)
        return avgEccentricity
