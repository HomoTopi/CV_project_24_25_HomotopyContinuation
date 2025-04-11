import numpy as np
from numpy import linalg as la
import logging
from HomoTopiContinuation.Rectifier.rectifier import Rectifier
from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography, Conic
import sympy as sp

class NumericRectifier(Rectifier):
    """
    A numeric implementation of the Rectifier using numpy for numerical computation.
    The algorithm is based on the intersection of conics method described in 
        "Jürgen Richter-Gebert Perspectives on Projective Geometry A Guided Tour Through Real and Complex Geometry".
        Chapter 11.1.  
    """

    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics using numerical intersection conic method.

        Args:
            C_img: A Conics object representing the image conics.
            
        """
        
        
        return
    

    def _solve_symbolic_cubic_equation(self, alpha: np.complex128, beta: np.complex128, gamma: np.complex128, delta: np.complex128) -> tuple[np.ndarray, np.ndarray]:
        # Define symbolic variables
        lam, mu = sp.symbols('lambda mu')
        
        # Create the homogeneous polynomial
        poly = alpha * lam**3 + beta * lam**2 * mu + gamma * lam * mu**2 + delta * mu**3
        
        # Solve by setting mu = 1 (dehomogenization) and finding roots
        cubic_eq = poly.subs(mu, 1)
        lambda_sols = sp.solve(cubic_eq, lam)
        
        # The solutions are pairs (lambda, 1)
        # Also add the solution at infinity if delta = 0
        solutions = [(sol, 1) for sol in lambda_sols]
        if alpha == 0:
            solutions.append((1, 0))  # Point at infinity
            
        print("solutions: ", solutions)
        
        # convert to correct format (all the lambdas together, all the mus together)
        lamb = np.array([sol[0] for sol in solutions])
        mu = np.array([sol[1] for sol in solutions])
        
        # convert to complex128
        lamb = lamb.astype(np.complex128)
        mu = mu.astype(np.complex128)
        
        print("lamb: ", lamb)
        print("mu: ", mu)
        
        return lamb, mu
    
    def _compute_cubic_equation(self, alpha: np.complex128, beta: np.complex128, gamma: np.complex128, delta: np.complex128) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the cubic equation.
        """
        
        M, L = self._get_M_L_matrix(alpha, beta, gamma, delta)
        w = np.complex64(-1/2 + (np.sqrt(3)/2 * 1j))
        
        W = np.array([
            [w, 1, w**2], 
            [1, 1, 1], 
            [w**2, 1, w]
        ])
        
        lamb = W @ L
        mu = W @ M
        
        
        self.logger.info("lamb: %s", lamb)
        self.logger.info("mu: %s", mu)
        
        
        return lamb,  mu

    def _get_M_L_matrix(self, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the M and L matrices for the cubic equation.
        """
        
        W = -2 * (beta**3) + 9 * alpha * beta * gamma - 27 * delta * (alpha**2)
        self.logger.info("W: %s", W)
        D = - (beta**2) * (gamma**2) + 4 * alpha * (gamma ** 3)  + 4 * (beta ** 3) * delta - 18 * alpha * beta * gamma * delta + 27 * (alpha**2) * (delta**3)
        self.logger.info("D: %s", D)
        Q = W - (alpha * np.sqrt(np.complex64(27*D)))
        self.logger.info("Q: %s", Q)
        R = (4*Q)**(1/3)
        self.logger.info("R: %s", R)
        L = np.array([[(2*(beta**2)) - (6 * alpha * gamma), -beta, R ]]).T
        #ones = np.ones(R.shape)
        #twos = 2 * ones.copy()
        M = 3 * alpha * np.array([[R,1,2]]).T
       
        self.logger.info("M: %s", M)
        
        assert (M.shape == (3,1))
        assert (L.shape == (3,1))
        
        return M, L
    
    def _compute_conics_intersection(self, A: Conic, B: Conic) -> np.ndarray:
        """
        Compute the intersection points of two conics. As described in Jürgen Richter-Gebert Perspectives on Projective Geometry A Guided Tour Through Real and Complex Geometry.
        
        Input:
            A: Conic object representing the first conic
            B: Conic object representing the second conic
            
        Output:
            intersection_points: numpy array of shape (n, 3) representing the intersection points of the two conics
        """
        # step 1: compute the parameters
        alpha, beta, gamma, delta = self._get_parameters(A, B)

        # step 2: solve the cubic equation
        #lamb, mu = self._compute_cubic_equation(alpha, beta, gamma, delta)
        lamb, mu = self._solve_symbolic_cubic_equation(alpha, beta, gamma, delta)
        if len(lamb) == 0:
             self.logger.error("Cubic equation solver returned no solutions.")
             raise ValueError("Cubic equation solver returned no solutions.")

        # compute the intersection conics (one for each lambda/mu pair)
        # Ensure broadcasting works correctly even if lamb, mu are scalars (though they should be arrays)
        lamb = np.atleast_1d(lamb)
        mu = np.atleast_1d(mu)
   
        result_Cs = lamb[:, np.newaxis, np.newaxis] * A.M + mu[:, np.newaxis, np.newaxis] * B.M
        
        # Iterate through each potential intersection conic C = lambda*A + mu*B
        for i in range(len(lamb)):
            C_mat = result_Cs[i, :, :]
            self.logger.debug("Processing solution %d: lambda=%s, mu=%s", i, lamb[i], mu[i])
            self.logger.debug("Result Conic Matrix C[%d]:\n%s", i, C_mat)

            # Ensure the resulting conic matrix is nearly singular
            det_C = la.det(C_mat)
            self.logger.debug("Determinant of Result Conic C[%d]: %s", i, det_C)

            # Check if determinant is close to zero. If not, this root might be spurious
            # due to numerical issues, or the original method was flawed.
            if np.abs(det_C) > self.treshold * 10: # Use a slightly larger tolerance for check
                 self.logger.warning("Determinant of C[%d] (%s) is not close to zero. Skipping this solution.", i, det_C)
                 
                 continue  # Skip to the next lambda/mu pair

            # If det_C is close enough to zero, proceed to split the conic
            C_degenerate = Conic(C_mat)
            try:
                # Step 3: Split the degenerate conic into two lines g and h
                g, h = self._split_conic(C_degenerate)
                #g, h = self._split_conic_svd(C_degenerate)
                self.logger.debug("Split lines for C[%d]: g=%s, h=%s", i, g, h)

                # Step 4: Intersect the lines g and h with one of the original conics (e.g., A)
                points_g = self._compute_conic_line_intersection(A, g)
                points_h = self._compute_conic_line_intersection(A, h)
                
                break # only one solution is needed
            
            except ValueError as e:
                self.logger.warning("Could not split conic C[%d] or find intersections: %s", i, e)

        intersection_points = np.concatenate((points_g, points_h), axis=0)
        assert intersection_points.shape == (4, 3), "The number of intersection points should be 4"
        
        # Combine all found intersection points
        if not intersection_points:
            self.logger.warning("No valid intersection points found after processing all solutions.")
            return np.zeros((0, 3), dtype=np.complex128)

        self.logger.info("Found %d raw intersection points:\n%s", intersection_points.shape[0], intersection_points)
        
        return intersection_points

    def _compute_adjugate(self, M: np.ndarray) -> np.ndarray:
        """
        Compute the adjugate matrix (transpose of the cofactor matrix) for a 3x3 matrix.
        Ref: https://en.wikipedia.org/wiki/Adjugate_matrix#3_%C3%97_3_generic_matrix
        
        
        
        """
        # Ensure input is numpy array and use complex128 for calculations
        M = np.asarray(M, dtype=np.complex128)

        cofactor_matrix = np.zeros_like(M, dtype=np.complex128)

        cofactor_matrix[0, 0] = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
        cofactor_matrix[0, 1] = -(M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
        cofactor_matrix[0, 2] = M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]

        cofactor_matrix[1, 0] = -(M[0, 1] * M[2, 2] - M[0, 2] * M[2, 1])
        cofactor_matrix[1, 1] = M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]
        cofactor_matrix[1, 2] = -(M[0, 0] * M[2, 1] - M[0, 1] * M[2, 0])

        cofactor_matrix[2, 0] = M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1] 
        cofactor_matrix[2, 1] = -(M[0, 0] * M[1, 2] - M[0, 2] * M[1, 0])
        cofactor_matrix[2, 2] = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

        adjugate_matrix = cofactor_matrix.T
        return adjugate_matrix

    def _split_conic(self, C: Conic) -> np.ndarray:
        """
        Split a degenerate conic (rank 2) into its two constituent lines.
        Implements the procedure from Jürgen Richter-Gebert Perspectives on Projective Geometry A Guided Tour Through Real and Complex Geometry describing splitting via the adjugate matrix.

        Args:
            C: A Conic object representing the degenerate conic (det(C.M) should be near zero).

        Returns:
            A tuple (g, h) containing the homogeneous coordinates of the two lines.
        """
        A = C.M.astype(np.complex64) # Work with complex numbers
        
        # Step 1: B := A△ (Adjugate of A)
        B_adj = self._compute_adjugate(A)
        self.logger.debug("Adjugate Matrix B:\n%s", B_adj)
        
        #### RANK 1 CASE ####
        # Check for rank 1 case (double line) where B_adj is zero
        if la.norm(B_adj) < self.treshold:
            self.logger.warning("Adjugate matrix is near zero. Assuming double line conic.")
            # Find a non-zero row/column in A
            norms = la.norm(A, axis=1)
            non_zero_row_idx = np.where(norms > self.treshold)[0]
            if len(non_zero_row_idx) == 0:
                raise ValueError("Input matrix A for splitting is zero or near-zero.")
            g = A[non_zero_row_idx[0], :]
            self.logger.debug("Found double line g: %s", g)

            lines = np.concatenate((g, g), axis=0)
            
            assert lines.shape == (2, 3), "The number of lines should be 2"
            
            return lines
        #### END OF THE RANK 1 CASE ####
        
        
        #### RANK 2 CASE ####
        # Step 2: Find index i of a non-zero diagonal entry of B_adj
        diag_B = np.diag(B_adj)
        non_zero_diag_indices = np.where(np.abs(diag_B) > self.treshold)[0]
        
        if len(non_zero_diag_indices) == 0:
             raise ValueError("No non-zero diagonal element found in Adjugate B")
        else:
            i = non_zero_diag_indices[0]
            Bii = diag_B[i]
            self.logger.debug("Using index i=%d from non-zero diagonal element B[%d,%d]=%s", i, i, i, Bii)


        # Step 3: β = sqrt(-Bi,i) (Since B = -pp^T implies Bii = -pi^2)
        beta_sq = -Bii
       
        beta = np.sqrt(beta_sq.astype(np.complex64))
        if np.abs(beta) < self.treshold:
                raise ValueError(f"Calculated beta is near zero (from B[{i},{i}]={Bii}). Cannot proceed.")

        # Step 4: p = Bi / β (where Bi is the i-th column of B_adj)
        Bi_col = B_adj[:, i]
        p = Bi_col / beta
        self.logger.debug("Calculated intersection point p = B[:,%d]/beta: %s", i, p)


        p = p.reshape(3, 1) # Ensure p is a column vector

        # Step 5: C_rank1 = A + Mp (or A - Mp)
        # Mp is the skew-symmetric matrix associated with p = (lambda, mu, tau)
        Mp = self._get_Ml(p.flatten())
        C_rank1 = A + Mp # Should be proportional to 2gh^T

        # Step 6: Find (row_idx, col_idx) of a non-zero element in C_rank1
        x, y = np.where(C_rank1 != 0)
        i, j = x[0], y[0]
        # Step 7: g is the row_idx-th row of C, h is the col_idx-th column of C
        # Normalization helps later steps, divide by the max element found
        g = C_rank1[i, :] #/ C_rank1[i, j]
        h = C_rank1[:, j] / C_rank1[i, j] # Use same normalization factor

        # Return as flat arrays
        g = g.flatten()
        h = h.flatten()
        
        lines = np.concatenate((g, h), axis=0)
        assert lines.shape == (2, 3), "The number of lines should be 2"
        #### END OF THE RANK 2 CASE ####
        
        return lines

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
        
        M_l = self._get_Ml(l)
        
        #1st step:
        
        B = M_l.T @ C.M @ M_l
        #2nd step:
        # TODO: search for the best submatrix (determinant is negative)
        #selected_index = self._get_best_submatrix(B)
        submatrix, det = self._get_best_submatrix(B)
 
        if det < 0:
            sqrt_det = np.sqrt(np.complex64(-det))
        else:
            sqrt_det = np.sqrt(-det)
        if l[0,2] > 0:
            alpha = (1/l[0,2]) * sqrt_det
        else:
            alpha = sqrt_det

        assert la.norm(alpha) > self.treshold, "The selected submatrix is singular"        
        
        #3rd step:
        
        C = B + (alpha * M_l)
           
        #4th step:
        non_zero_indices = np.nonzero(C)
        i, j = non_zero_indices[0][0], non_zero_indices[1][0]
        # the intersection points are
        p = C[i,:]
        q = C[:,j]
        
        # set to zero the elements with a norm under a threshold
        p = p * (la.norm(p, axis=0) > self.treshold)
        q = q * (la.norm(q, axis=0) > self.treshold)
                
        #if p[2] > self.treshold:
        #    p = p / p[2]
        #if q[2] > self.treshold:00
        #    q = q / q[2]
        return np.array([p,q])
    
    def _get_Ml(self, l: np.ndarray) -> np.ndarray:
        """
        Get the skew-symmetric matrix M_l for a given line l = (lambda, mu, tau).
        M_l =
        [[  0, tau, -mu],
         [-tau,   0, lam],
         [  mu,-lam,   0]]
        """
        # Ensure l is a flat array/vector
        l = l.flatten()
        assert l.shape == (3,), f"Line must be a 3-element vector, got shape {l.shape}"

        lam, mu, tau = l[0], l[1], l[2]

        # Use the dtype of the input line
        return np.array([
            [0, tau, -mu],
            [-tau, 0, lam],
            [mu, -lam, 0]
        ], dtype=l.dtype)


    def _get_best_submatrix(self, B: np.ndarray) -> tuple[np.ndarray, np.complex128]:
        """
        Get the best 2x2 submatrix of B for line-conic intersection.
        "Best" usually means one with a negative determinant for real intersections,
        but for complex intersections, any non-zero determinant submatrix might work.
        This implementation prioritizes negative determinants, then positive.
        Returns the submatrix and its determinant.
        """
        best_submatrix = None
        best_det = None
        found_neg = False

        # Ensure B is complex128
        B = B.astype(np.complex128)

        for i in range(B.shape[0] - 1):
            for j in range(B.shape[1] - 1):
                # Extract submatrix indices carefully for 3x3 B
                rows = np.delete(np.arange(3), i)
                cols = np.delete(np.arange(3), j)
                submatrix = B[np.ix_(rows, cols)] # Use np.ix_ for proper submatrix extraction

                # Check if submatrix is 2x2 (should be if B is 3x3)
                if submatrix.shape != (2, 2):
                    continue

                det = la.det(submatrix)

                # Check for non-zero determinant
                if np.abs(det) > self.treshold:
                    # Prioritize negative real part (heuristic for original intent)
                    if det.real < -self.treshold and not found_neg:
                         best_submatrix = submatrix
                         best_det = det
                         found_neg = True
                    # Otherwise, store the first non-zero one found
                    elif best_submatrix is None:
                         best_submatrix = submatrix
                         best_det = det

        if best_submatrix is not None:
            self.logger.debug("Selected best submatrix with determinant: %s", best_det)
            return best_submatrix, best_det
        else:
            self.logger.error("No non-singular 2x2 submatrix found in B:\n%s", B)
            raise ValueError("No non-singular 2x2 submatrix found for line-conic intersection.")

    def _get_parameters(self, A: Conic, B: Conic) -> np.ndarray:
        """
        Compute the parameters alpha, beta, gamma, delta of the equation given by:
        det(lambda * A + mu * B) = 0
        
        """
        alpha = la.det(A.M)
        beta  = la.det(np.array([A.M[:,0], A.M[:,1], B.M[:,2]])) + la.det(np.array([A.M[:,0], B.M[:,1], A.M[:,2]])) + la.det(np.array([B.M[:,0], A.M[:,1], A.M[:,2]]))
        gamma = la.det(np.array([A.M[:,0], B.M[:,1], B.M[:,2]])) + la.det(np.array([B.M[:,0], A.M[:,1], B.M[:,2]])) + la.det(np.array([B.M[:,0], B.M[:,1], A.M[:,2]]))
        delta = la.det(B.M)
        
        self.logger.info("alpha: %s, beta: %s, gamma: %s, delta: %s", alpha, beta, gamma, delta)
        
        return alpha, beta, gamma, delta
        
        
        
        
        