from abc import ABC, abstractmethod
from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography
import logging
import numpy as np
import numpy.linalg as la

class Rectifier(ABC):
    """
    Abstract base class for rectification algorithms.

    This class defines the interface that all rectifiers must implement.
    """
    def __init__(self, treshold: float = 1e-3):
        self.treshold = treshold
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

    def _compute_h_from_svd(self, imDCCP: np.ndarray) -> Homography:
        """
        Compute the Homography from the SVD of the image dual conic.
        """
        eigs = np.linalg.eigvals(imDCCP)

        # thresholding
        eigs[np.abs(eigs) < self.treshold] = 0

        if np.any(eigs < 0):
            self.logger.error("imDCCP is not positive definite")
            raise ValueError(
                f"imDCCP is not positive definite! No homography can be computed. eigs: {eigs}")

        self.logger.info(f"imDCCP\n: {imDCCP}")

        # Singular value decomposition
        U, S, Vt = la.svd(imDCCP)
        self.logger.info(f"U\n: {U}")
        self.logger.info(f"S\n: {S}")
        self.logger.info(f"V\n: {Vt}")

        # Compute the homography
        H = np.diag(1.0 / np.sqrt([S[0], S[1], 1.0])) @ U.T

        self.logger.info(f"H: {H}")

        return Homography(H)