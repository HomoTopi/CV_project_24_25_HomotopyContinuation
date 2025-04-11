import numpy as np
import json

from HomoTopiContinuation.DataStructures.datastructures import Conics, Img, Homography

class ConicWarper:
    """
    Class for warping conics using the recontructed homography matrix.
    
    This class takes the reconstructed homography matrix and applies it to conics to warp them.
    """
    def warpConics(self):
        """
        Warps the conics using the reconstructed homography matrix.
        """
        with open("src\\HomoTopiContinuation\\Data\\sceneImage.json", "r") as file:
            data = json.load(file)

        with open("src\\HomoTopiContinuation\\Data\\rectifiedHomography.json", "r") as file:
            rectified_homographies = json.load(file)

        rectified_conics = []

        # Apply corresponding homography to each set of conics
        for img_json, H_json in zip(data, rectified_homographies):
            img = Img.from_json(img_json)
            C_img = img.C_img
            H = Homography.from_json(H_json)

            if H is not None:
                C1_warped = C_img.C1.applyHomography(H)
                C2_warped = C_img.C2.applyHomography(H)
                C3_warped = C_img.C3.applyHomography(H)
                rectified_conics.append(Conics(C1_warped, C2_warped, C3_warped))

        with open('src\\HomoTopiContinuation\\Data\\rectifiedConics.json', 'w') as file:
            json.dump([conic.to_json() for conic in rectified_conics], file, indent=4)

    
if __name__ == "__main__":
    conic_warper = ConicWarper()
    conic_warper.warpConics()