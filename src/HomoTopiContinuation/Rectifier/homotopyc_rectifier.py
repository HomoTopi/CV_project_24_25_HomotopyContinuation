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
                
        # call the __init__ of the parent class
        super().__init__()
        
    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics using SymPy.

        Args:
            C_img (Conics): A pair of conics in the image

        Returns:
            Homography: The rectification homography
        """
        
        script = self._get_script('Julia/rectify.jl')
        self.logger.info("Rectifying using HomotopyContinuation.jl")
        self.logger.debug(f"Conics: {C_img}")
        
        self.logger.info(f"Evaluating script: {script}")

        # Extract algebraic parameters from conics
        a1, b1, c1, d1, e1, f1 = C_img.C1.to_algebraic_form()
        a2, b2, c2, d2, e2, f2 = C_img.C2.to_algebraic_form()
        a3, b3, c3, d3, e3, f3 = C_img.C3.to_algebraic_form()
        
        jl.a1, jl.b1, jl.c1, jl.d1, jl.e1, jl.f1 = np.array([a1, b1, c1, d1, e1, f1])
        jl.a2, jl.b2, jl.c2, jl.d2, jl.e2, jl.f2 = np.array([a2, b2, c2, d2, e2, f2])
        jl.a3, jl.b3, jl.c3, jl.d3, jl.e3, jl.f3 = np.array([a3, b3, c3, d3, e3, f3])
        
        jl.seval(script)
        ## TODO: check if casting is the same as the one used in SymPy
        solutions = np.array([ np.complex128(sol) for sol in jl.sol])
        #real_solutions = np.array([ np.float64(sol) for sol in jl.real_sol])
        self.logger.info(f"Result: {solutions}")
        #self.logger.info(f"Real solutions: {real_solutions}")
        ## TODO: extract the intersection points from the result
        ## TODO: compute the rectification homography both by svd and by a fully homotopy continuation approach
        
        assert len(solutions) >= 2, "No intersection points found"
        
        II = solutions[0]
        JJ = solutions[1]
        self.logger.info(f"II: {II}")
        self.logger.info(f"JJ: {JJ}")
        
        # Compute the dual conic of the circular points
        imDCCP = np.outer(II, JJ) + np.outer(JJ, II)

        # Normalize
        imDCCP = imDCCP / la.norm(imDCCP)
        self.logger.info(f"imDCCP: {imDCCP}")

        H = self._compute_h_from_svd(imDCCP)

        return H
        
    
    def _get_script(self, filename: str) -> str:
        path = os.path.join(os.path.dirname(__file__), filename)
        script = open(path, "r").read()
        return script
