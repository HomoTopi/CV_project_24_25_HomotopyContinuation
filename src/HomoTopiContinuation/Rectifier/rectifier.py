from abc import ABC, abstractmethod
from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography
import logging

class Rectifier(ABC):
    """
    Abstract base class for rectification algorithms.

    This class defines the interface that all rectifiers must implement.
    """
    def __init__(self):
        logging.basicConfig(
            filename='rectifier.log',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(self.__class__.__name__)

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
