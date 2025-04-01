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
        
        assert imDCCP.shape == (3, 3), "imDCCP must be a 3x3 matrix"

        self.logger.info(f"imDCCP\n: {imDCCP}")

        # Singular value decomposition
        U, S, Vt = la.svd(imDCCP)
        if np.any(S < 0):
            self.logger.error("imDCCP is not positive definite")
            raise ValueError(
                f"imDCCP is not positive definite! No homography can be computed. Singular values: {S}")

        self.logger.info(f"U\n: {U}")
        self.logger.info(f"S\n: {S}")
        self.logger.info(f"V\n: {Vt}")

        # Compute the homography
        H = np.diag(1.0 / np.sqrt([S[0], S[1], 1.0])) @ U.T

        self.logger.info(f"H: {H}")

        return Homography(H)

    
    def compute_imDCCP_from_solutions(self, sols: np.ndarray) -> np.ndarray:
        """
        Compute the image dual conic from the solutions of the conic equations.
        """
        
        assert len(sols.shape) >= 2, "Solutions must be at least 2!"
        # remove all points which have all real parts
        # [
        # [ x_1, y_1, w_1],
        # [ x_2, y_2, w_2],
        # ]
        # if all x_i, y_i, w_i are real, remove the point
        sols = sols[~np.all(np.isreal(sols), axis=1)]
        self.logger.info(f"Complex solutions after filtering: {sols}")

        # if there are less than 2 complex solutions, raise an error
        if len(sols) < 2:
            self.logger.error("Less than 2 complex solutions found")
            raise ValueError(
                f"Less than 2 complex solutions found! sols: {sols}")

        # Extract intersection points
        II = sols[0][:, None]
        JJ = sols[1][:, None]

        self.logger.info(f"II: {II}")
        self.logger.info(f"JJ: {JJ}")

        # Compute the dual conic of the circular points
        imDCCP = II @ JJ.T + JJ @ II.T
        
        assert imDCCP.shape == (3, 3), "imDCCP must be a 3x3 matrix"
        
        return imDCCP
        
        
        
        
        
        
        
