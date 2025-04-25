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
        M = conic.M
        # Normalize the conic matrix
        if (conic.M[2, 2] != 0):
            conic.M = conic.M / conic.M[2, 2]

        # Extract coefficients
        A = M[0, 0]
        B = 2*M[0, 1]
        C = M[1, 1]
        D = 2*M[0, 2]
        E = 2*M[1, 2]
        F = M[2, 2]

        # Quadratic form and linear terms
        Q = np.array([[A, B/2.], [B/2., C]])
        L = np.array([D, E])

        # Check for parabola: det(Q) == 0
        detQ = np.linalg.det(Q)
        if np.isclose(detQ, 0):
            return 1.0  # Parabola eccentricity

        # Compute center of the conic: solve 2Q c + L = 0 => Q c = -L/2
        center = -0.5 * np.linalg.inv(Q) @ L

        # Compute constant term after translation
        F0 = center @ Q @ center + L @ center + F

        # Eigenvalues of Q
        evals, _ = np.linalg.eig(Q)
        # Denominators for canonical form: lambda_i x_i^2 = -F0
        denoms = -F0 / evals

        # Filter only real denominators
        if not np.all(np.isreal(denoms)):
            raise ValueError("Conic parameters lead to non-real axis lengths.")
        denoms = denoms.real

        # Identify type
        positive = denoms > 0
        if positive.sum() == 2:
            # Ellipse
            a2, b2 = np.sort(denoms)[::-1]
            if b2 <= 0:
                raise ValueError("Invalid ellipse axes.")
            e = np.sqrt(1 - b2/a2)
        elif positive.sum() == 1:
            # Hyperbola
            # a^2 is the positive denom, b^2 is abs(negative denom)
            a2 = denoms[positive][0]
            b2 = abs(denoms[~positive][0])
            e = np.sqrt(1 + b2/a2)
        else:
            return float('inf')  # Degenerate conic

        return float(e)

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
