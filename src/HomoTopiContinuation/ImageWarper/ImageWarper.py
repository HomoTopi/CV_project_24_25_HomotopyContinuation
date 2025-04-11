import numpy as np
import cv2
from HomoTopiContinuation.DataStructures.datastructures import Homography


class ImageWarper:
    """
    This class is used to warp an image using a homography matrix.
    """

    def __call__(self, img: np.ndarray, H: Homography) -> np.ndarray:
        """
        Warp an image using a homography matrix.

        Args:
            img (numpy.ndarray): The image to warp
            H (Homography): The homography matrix

        Returns:
            np.ndarray: The warped image
        """
        # Get the shape of the image
        h, w = img.shape[:2]

        # Warp the image
        warped_img = cv2.warpPerspective(img, H.H, (w, h))

        return warped_img
