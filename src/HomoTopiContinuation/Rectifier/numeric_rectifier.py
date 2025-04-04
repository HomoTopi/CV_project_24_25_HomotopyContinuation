import numpy as np
from numpy import linalg as la
import logging
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
    
    def _compute_conics_intersection(self, A: Conic, B: Conic) -> np.ndarray:
        """
        Compute the intersection points of two conics.
        """
        # step 1: compute the parameters
        alpha, beta, gamma, delta = self._get_parameters(A, B)
        
        # step 2: solve the cubic equation
        
        M, L = self._get_M_L(alpha, beta, gamma, delta)
        
        print(M)
        print(L)
        
        w = np.complex64(-1/2 + np.sqrt(np.complex64(3/4)) * 1j)
        
        W = np.array([
            [w, 1, w**2], 
            [1, 1, 1], 
            [w**2, 1, w]
        ])
        
        lamb = la.solve(W, L)
        mu = la.solve(W,M)
        print("lamb: ", lamb)
        print("mu: ", mu)
        
        # compute the intersection conic (those are 3 conics, all representing the solution)
        result_Cs = lamb[:, np.newaxis] * A.M + mu[:, np.newaxis] * B.M
        
        # select the first C
        C = result_Cs[0,:,:]
        print("C: \n", C)
        C = Conic(C)
        i, j = C.get_non_zero_entry()
        
        g = C.M[i,:]
        h = C.M[:,j]
        
        
        print("g: ", g)
        print("h: ", h)
        
        return g, h
        
    
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

        #assert l[0,2] != 0, "input line lies at the infinity!"
        
        M_l = self._get_M_l(l)
        
        #1st step:
        
        B = M_l.T @ C.M @ M_l

        #2nd step:
        # TODO: search for the best submatrix (determinant is negative)
        #selected_index = self._get_best_submatrix(B)
        submatrix, det = self._get_best_submatrix(B)
 
        if det > 0:
            sqrt_det = (1j * np.sqrt(abs(det))) 
        else:
            sqrt_det = np.sqrt(-det)
            
        if l[0,2] > 0:
            alpha = (1/l[0,2]) * sqrt_det
        else:
            alpha = sqrt_det

        print(alpha)
        assert np.linalg.norm(alpha) > self.treshold, "The selected submatrix is singular"        
        
        #3rd step:
        
        C = B + (alpha * M_l)
        print("CCCC: \n", C)
        C = Conic(C)        
        #4th step:
        i, j = C.get_non_zero_entry()
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
        [mu -lambda 0]
        """
        
        assert l.shape == (1, 3), "Line must be a 1x3 numpy array"
       
        tau = l[0,2]
        mu = l[0,1]
        lam = l[0,0]

        return np.array([
            [0, tau, -mu],
            [-tau, 0, lam],
            [mu, -lam, 0]
        ])

    def _get_best_submatrix(self, B: np.ndarray) -> np.ndarray:
        """
        Get the best submatrix of B.
        """
        
        # get the first submatrix 2x2 with a negative determinant
        positive_det = False
        for i in range(B.shape[0] - 1):
            for j in range(B.shape[1] - 1):
                submatrix = B[i:i+2, j:j+2]
                det = la.det(submatrix)
                if det < 0:
                    return submatrix, det
                elif det > 0 and not positive_det:
                    positive_det = True
                    positive_submatrix = submatrix
                    positive_det = det
        
        if positive_det:
            return positive_submatrix, positive_det
        else:
            raise ValueError("No submatrix with a negative determinant found")
        
    def _get_parameters(self, A: Conic, B: Conic) -> np.ndarray:
        """
        Compute the parameters alpha, beta, gamma, delta of the equation given by:
        det(lambda * A + mu * B) = 0
        """
        
        alpha = la.det(np.array([A.M[:,0], A.M[:,1], A.M[:,2]]))
        beta  = la.det(np.array([A.M[:,0], A.M[:,1], B.M[:,2]])) + la.det(np.array([A.M[:,0], B.M[:,1], A.M[:,2]])) + la.det(np.array([B.M[:,0], A.M[:,1], A.M[:,2]]))
        gamma = la.det(np.array([A.M[:,0], B.M[:,1], B.M[:,2]])) + la.det(np.array([B.M[:,0], A.M[:,1], B.M[:,2]])) + la.det(np.array([B.M[:,0], B.M[:,1], A.M[:,2]]))
        delta = la.det(np.array([B.M[:,0], B.M[:,1], B.M[:,2]]))
        
        self.logger.info("alpha: %s, beta: %s, gamma: %s, delta: %s", alpha, beta, gamma, delta)
        
        return alpha, beta, gamma, delta
    
    def _get_M_L(self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Get the M and L matrices for the cubic equation.
        """
        
        W = -2 * beta**3 + 9 * alpha * beta * gamma - 27 * delta * alpha**2
        self.logger.info("W: %s", W)
        D = - beta**2 * gamma**2 + 4 * alpha * gamma ** 3  + 4 * beta ** 3 * delta - 18 * alpha * beta * gamma * delta + 27 * alpha**2 * delta**3
        self.logger.info("D: %s", D)
        Q = W - (alpha * np.sqrt(np.complex64(27*D)))
        self.logger.info("Q: %s", Q)
        R = (4*Q)**(1/3)
        self.logger.info("R: %s", R)
        L = np.array([[2*beta**2 - 6 * alpha * gamma, - beta, R ]]).T
        ones = np.ones(R.shape)
        twos = 2 * ones.copy()
        M = 3 * alpha * np.array([[R,ones.T,twos.T]]).T
        self.logger.info("M: %s", M)
 
        return M, L
        
        
        
        
        