import numpy as np
import logging
import os
import requests
import json

from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography
from .rectifier import Rectifier


class HomotopyContinuationRectifier(Rectifier):
    """
    A standard implementation of the Rectifier using SymPy for symbolic computation.

    This rectifier solves a system of conic equations symbolically to find intersection
    points, then computes the homography.
    """

    def __init__(self):
        """
        Initialize the rectifier and set up the Julia environment.
        """
        DOMAIN = "localhost"
        PORT = 8081
        SERVICE = "rectify"

        self.service_url = f"http://{DOMAIN}:{PORT}/{SERVICE}"
        super().__init__()

    def computeImagesOfCircularPoints(self, C_img: Conics) -> np.ndarray:
        # Extract algebraic parameters from conics
        a1, b1, c1, d1, e1, f1 = C_img.C1.to_algebraic_form()
        a2, b2, c2, d2, e2, f2 = C_img.C2.to_algebraic_form()
        a3, b3, c3, d3, e3, f3 = C_img.C3.to_algebraic_form()

        self.logger.info(
            f"Equation 1: {a1}*x^2 + {b1}*x*y + {c1}*y^2 + {d1}*x*w + {e1}*y*w + {f1}*w^2")
        self.logger.info(
            f"Equation 2: {a2}*x^2 + {b2}*x*y + {c2}*y^2 + {d2}*x*w + {e2}*y*w + {f2}*w^2")
        self.logger.info(
            f"Equation 3: {a3}*x^2 + {b3}*x*y + {c3}*y^2 + {d3}*x*w + {e3}*y*w + {f3}*w^2")

        data = {
            "conics": [
                [a1, b1, c1, d1, e1, f1],
                [a2, b2, c2, d2, e2, f2],
                [a3, b3, c3, d3, e3, f3]
            ]
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(
            self.service_url, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise Exception(f"API call failed: {response.json()}")

        solutions = response.json()["complex_sols"]

        self.logger.info(f"Solutions: {solutions}")
        # TODO: check if casting is the same as the one used in SymPy
        solutions = np.array([[complex(sol["re"], sol["im"]) for sol in sol_tuple]
                              for sol_tuple in solutions])

        self.logger.info(f"Result: {solutions}")
        return solutions

    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics using the Julia Homotopy Continuation package.

        Returns:
            Homography: The rectification homography
        """

        self.logger.info("Rectifying using HomotopyContinuation.jl")
        self.logger.debug(f"Conics: {C_img}")

        # Compute the images of the circular points
        solutions = self.computeImagesOfCircularPoints(C_img)

        imDCCP = self.compute_imDCCP_from_solutions(solutions)
        H = self._compute_h_from_svd(imDCCP)

        return H


if __name__ == "__main__":
    rectifier = HomotopyContinuationRectifier()
    rectifier.rectify("../data/")
