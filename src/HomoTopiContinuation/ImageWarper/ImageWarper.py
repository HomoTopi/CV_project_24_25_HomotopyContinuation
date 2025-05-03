import numpy as np
import cv2
from HomoTopiContinuation.DataStructures.datastructures import Homography


class ImageWarper:
    """
    This class is used to warp an image using a homography matrix.
    """

    def __call__(self, img: np.ndarray, H: Homography) -> np.ndarray:
        """
        Warp an image using a homography matrix and compute the shape of the final result.

        Args:
            img (numpy.ndarray): The image to warp
            H (Homography): The homography matrix

        Returns:
            np.ndarray: The warped image
        """
        # Get the shape of the image
        h, w = img.shape[:2]

        # Define the four corners of the image
        corners = np.array([
            [0, 0, 1],        # Top-left corner
            [w - 1, 0, 1],    # Top-right corner
            [w - 1, h - 1, 1],  # Bottom-right corner
            [0, h - 1, 1]     # Bottom-left corner
        ]).T  # Transpose to make it 3x4 for matrix multiplication

        # Transform the corners using the homography matrix
        transformed_corners = H.H @ corners

        # Normalize the transformed corners (convert from homogeneous to Cartesian coordinates)
        transformed_corners /= transformed_corners[2]

        # Extract the x and y coordinates
        x_coords = transformed_corners[0]
        y_coords = transformed_corners[1]

        # Compute the bounding box of the transformed image
        min_x, max_x = int(np.floor(x_coords.min())), int(
            np.ceil(x_coords.max()))
        min_y, max_y = int(np.floor(y_coords.min())), int(
            np.ceil(y_coords.max()))

        # Compute the size of the output image
        new_width = max_x - min_x
        new_height = max_y - min_y

        # Adjust the translation to ensure the warped image fits in the new canvas
        translation_matrix = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])

        # Combine the translation with the homography
        adjusted_homography = translation_matrix @ H.H

        # Warp the image with the adjusted homography
        warped_img = cv2.warpPerspective(
            img, adjusted_homography, (new_width, new_height))

        return warped_img, adjusted_homography
