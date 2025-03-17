from abc import ABC, abstractmethod
from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography


class Rectifier(ABC):
    """
    Abstract base class for rectification algorithms.

    This class defines the interface that all rectifiers must implement.
    """

    @abstractmethod
    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics to recover the homography.

        Args:
            C_img (Conics): A pair of conics in the image

        Returns:
            Homography: The rectification homography

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError(
            "Abstract method must be implemented by subclasses")
