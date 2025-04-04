import numpy as np
from numpy import linalg as la
import logging
from sympy import symbols, Eq, solve
from HomoTopiContinuation.Rectifier.rectifier import Rectifier
from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography, Conic

class NumericRectifier(Rectifier):
    """
    A numeric implementation of the Rectifier using numpy for numerical computation.
    """

    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics using numerical intersection conic method.

        Args:
            C_img: A Conics object representing the image conics.
            
        """
        
        
        return
    
    
    def _compute_conic_line_intersection(self, C: Conic, l: np.ndarray) -> np.ndarray:
        """
        Compute the intersection points of a conic and a line.

        Args:
            C: A Conic object representing the conic.
            l: A numpy array representing the line.
        """
        if l.shape == (3,):
            l = l.reshape(1, 3)
        
        assert  l.shape == (1, 3), "Line must be a 1x3 numpy array"

        assert l[0,2] != 0, "input line lies at the infinity!"
        
        M_l = self._get_M_l(l)
        
        #1st step:
        
        B = M_l.T @ C.M @ M_l

        #2nd step:
        # TODO: search for the best submatrix (determinant is negative)
        #selected_index = self._get_best_submatrix(B)
        
    
        alpha = (1/l[0,2]) * np.sqrt(-np.linalg.det(B[0:2, 0:2]))

        assert alpha > self.treshold, "The selected submatrix is singular"        
        
        #3rd step:
        
        C = B + (alpha * M_l)        
        
        #4th step:
        # get the first non-zero entry of C
        non_zero_entry = np.nonzero(C)
        if non_zero_entry:
            i, j = non_zero_entry[0][0], non_zero_entry[1][0]
        else:
            raise ValueError("No non-zero entry found in the matrix")
        
        # the intersection points are
        p = C[i,:]
        q = C[:,j]
        
        return np.array([p, q])
    
    def _get_M_l(self, l: np.ndarray) -> np.ndarray:
        """
        Get the M_l matrix for a given line.
        Given l = (lambda, mu, tau).T, the M_l matrix is defined as:
        M_l = 
        [0 tau -mu]
        [-tau 0 lambda]
        [-mu -lambda 0]
        """
        
        assert l.shape == (1, 3), "Line must be a 1x3 numpy array"
       
        tau = l[0,2]
        mu = l[0,1]
        lam = l[0,0]

        return np.array([
            [0, tau, -mu],
            [-tau, 0, lam],
            [-mu, -lam, 0]
        ])

    def _get_best_submatrix(self, B: np.ndarray) -> np.ndarray:
        """
        Get the best submatrix of B.
        """
        # TODO: implement the best submatrix selection
        return np.array([[0,1],[0,1]])
