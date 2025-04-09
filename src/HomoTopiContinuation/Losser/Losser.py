import numpy as np
from HomoTopiContinuation.DataStructures.datastructures import Homography, Conics, SceneDescription
from abc import ABC, abstractmethod


class Losser(ABC):
    """
    Abstract base class for a Losser."
    A losser is able to compute the quality of a rectification homography through the method compoteLoss.
    """

    @abstractmethod
    def computeLoss(h_true: Homography, h_computed: Homography) -> float:
        """
        Compute the loss between two homographies.

        Args:
            h_true (Homography): The true homography
            h_computed (Homography): The computed homography

        Returns:
            float: The loss between the two homographies
        """
        raise NotImplementedError(
            "Abstract method must be implemented by subclasses")

    def __call__(self, h_true: Homography, h_computed: Homography):
        """
        Compute the loss between two homographies using the computeLoss method.
        """
        return self.computeLoss(h_true, h_computed)


class SceneLosser(Losser):
    """
    Abstract base class for a losser that computes the loss based on the reconstructed objects.
    """

    @abstractmethod
    def computeCircleLoss(sceneDescription: SceneDescription, C_computed: Conics) -> float:
        """
        Computes the loss of a shape reconstruction based on the reconstracted circles.

        Args:
            sceneDescription (SceneDescription): The scene description
            C_computed (Conics): The reconstructed Circles

        Returns:
            float: The loss of the shape reconstruction
        """
        raise NotImplementedError(
            "Abstract method must be implemented by subclasses")

    def __call__(self, sceneDescription: SceneDescription, C_computed: Conics):
        """
        Compute the loss of a shape reconstruction based on the reconstracted circles using the computeCircleLoss method.
        """
        return self.computeCircleLoss(sceneDescription, C_computed)
