import numpy as np
from numpy import linalg as la
import logging
from juliacall import Main as jl
import juliapkg
import os


from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography
from .rectifier import Rectifier


class HomotopyContinuationRectifier(Rectifier):
    """
    A standard implementation of the Rectifier using SymPy for symbolic computation.

    This rectifier solves a system of conic equations symbolically to find intersection
    points, then computes the homography.
    """
    
    def __init__(self):
        # add the HomotopyContinuation package
        juliapkg.add("HomotopyContinuation","f213a82b-91d6-5c5d-acf7-10f1c761b327")
        juliapkg.resolve()
        
        logging.basicConfig(level=logging.DEBUG)
        
        # call the __init__ of the parent class
        super().__init__()

    def test_julia(self):
        # get script to eval from file
        path = os.path.join(os.path.dirname(__file__), 'Julia/test.jl')
        script = open(path, "r").read()
        jl.my_vec = np.array([1, 2, 3])
        jl.seval(script)
                
        
        
    
    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics using SymPy.

        Args:
            C_img (Conics): A pair of conics in the image

        Returns:
            Homography: The rectification homography
        """
        
        script = self._get_script('Julia/rectify.jl')
        logging.info("Rectifying using HomotopyContinuation.jl")
        logging.debug(f"Conics: {C_img}")
        
        logging.info(f"Evaluating script: {script}")

        # Extract algebraic parameters from conics
        a1, b1, c1, d1, e1, f1 = C_img.C1.to_algebraic_form()
        a2, b2, c2, d2, e2, f2 = C_img.C2.to_algebraic_form()
        
        
        jl.a1, jl.b1, jl.c1, jl.d1, jl.e1, jl.f1 = np.array([a1, b1, c1, d1, e1, f1])
        jl.a2, jl.b2, jl.c2, jl.d2, jl.e2, jl.f2 = np.array([a2, b2, c2, d2, e2, f2])
        
        jl.seval(script)
        
        result = jl.result
        logging.info(f"Result: {result}")
        
        ## TODO: extract the intersection points from the result
        ## TODO: compute the rectification homography both by svd and by a fully homotopy continuation approach
    
        return 

    def _get_script(self, filename: str) -> str:
        path = os.path.join(os.path.dirname(__file__), filename)
        script = open(path, "r").read()
        return script
