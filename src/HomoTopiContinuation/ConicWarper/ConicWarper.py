import numpy as np
import json

from HomoTopiContinuation.DataStructures.datastructures import Conics, Img, Homography

class ConicWarper:
    """
    Class for warping conics using the recontructed homography matrix.
    
    This class takes the reconstructed homography matrix and applies it to conics to warp them.
    """

    def warpConics(self, C_img : Conics, H : Homography) -> Conics:
        """
        Warps the conics using the reconstructed homography matrix.
        """
        # Apply corresponding homography to each set of conics
        C1_warped = C_img.C1.applyHomography(H)
        C2_warped = C_img.C2.applyHomography(H)
        C3_warped = C_img.C3.applyHomography(H)
        return Conics(C1_warped, C2_warped, C3_warped)


    
if __name__ == "__main__":
    conic_warper = ConicWarper()
    # Assuming C_img and H are already defined
    # C_img = Conics(...)  # Replace with actual conics
    # H = Homography(...)  # Replace with actual homography
    # conic_warper.warpConics(C_img, H)
    # For demonstration, we will use dummy data
    C_img = Conics()
    H = Homography()
    # Call the warpConics method
    warped_conics = conic_warper.warpConics(C_img, H)
    # Print the warped conics
    print(warped_conics)