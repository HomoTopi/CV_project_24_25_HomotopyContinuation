import numpy as np
import sympy as sp
from numpy import linalg as la
import logging

from DataStructures.datastructures import Conics, Homography
from .rectifier import Rectifier


class StandardRectifier(Rectifier):
    """
    A standard implementation of the Rectifier using SymPy for symbolic computation.
    
    This rectifier solves a system of conic equations symbolically to find intersection
    points, then computes the homography.
    """
    
    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics using SymPy.
        
        Args:
            C_img (Conics): A pair of conics in the image
            
        Returns:
            Homography: The rectification homography
        """
        logging.info("Rectifying using SymPy")
        
        logging.debug(f"Conics: {C_img}")
        
        # Extract algebraic parameters from conics
        a1, b1, c1, d1, e1, f1 = C_img.C1.to_algebraic_form()
        a2, b2, c2, d2, e2, f2 = C_img.C2.to_algebraic_form()
        
        logging.info(f"Equation 1: {a1}*x^2 + {b1}*x*y + {c1}*y^2 + {d1}*x + {e1}*y + {f1}")
        logging.info(f"Equation 2: {a2}*x^2 + {b2}*x*y + {c2}*y^2 + {d2}*x + {e2}*y + {f2}")
        
        # Define symbolic variables
        x, y = sp.symbols('x y')
        
        # Define equations
        eq1 = sp.Eq(a1*x**2 + b1*x*y + c1*y**2 + d1*x + e1*y + f1, 0)
        eq2 = sp.Eq(a2*x**2 + b2*x*y + c2*y**2 + d2*x + e2*y + f2, 0)
        
        # Solve the system of equations
        solutions = sp.solve((eq1, eq2), (x, y))
        logging.debug(f"Solutions: {solutions}")
        
        # Extract intersection points
        II = np.array([float(solutions[0][0]), float(solutions[0][1]), 1.0])
        JJ = np.array([float(solutions[1][0]), float(solutions[1][1]), 1.0])
        
        logging.debug(f"II: {II}")
        logging.debug(f"JJ: {JJ}")
        
        # Compute the dual conic of the circular points
        imDCCP = np.outer(II, JJ) + np.outer(JJ, II)
        
        # Normalize
        imDCCP = imDCCP / la.norm(imDCCP)
        logging.debug(f"imDCCP: {imDCCP}")
        
        # Singular value decomposition
        U, S, Vt = la.svd(imDCCP)
        logging.debug(f"U: {U}")
        logging.debug(f"S: {S}")
        logging.debug(f"V: {Vt}")
        
        # Compute the homography
        H = np.diag(1.0 / np.sqrt(S)) @ U.T
        
        logging.debug(f"H: {H}")
        
        return Homography(H) 