import numpy as np
from numpy import linalg as la
import logging
from sympy import symbols, Eq, solve


from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography
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
        self.logger.info("Rectifying using scipy")

        self.logger.info(f"Conics: {C_img}")

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

        x, y, w = symbols('x y w')
        eq = [Eq(a1*x**2 + b1*x*y + c1*y**2 + d1*x*w + e1*y*w + f1*w**2, 0),
              Eq(a2*x**2 + b2*x*y + c2*y**2 + d2*x*w + e2*y*w + f2*w**2, 0),
              Eq(a3*x**2 + b3*x*y + c3*y**2 + d3*x*w + e3*y*w + f3*w**2, 0)]

        # Solve the system of equations
        # TODO filter real solutions
        
        solutions = solve(eq, (x, y, w))
        self.logger.debug(f"Result of solve: {solutions}")

        # Extract complex solutions
        sols = np.array([[complex(expr.subs({x: 1, y: 1, w: 1})) for expr in expr_tuple] for expr_tuple in solutions])
        self.logger.info(f"Solutions before filtering: {sols}")

        if len(sols) == 0:
            self.logger.error("No solutions found")
            raise ValueError(
                f"No solutions found! sols: {sols}")

        # remove all points which have all real parts
        # [
        # [ x_1, y_1, w_1],
        # [ x_2, y_2, w_2],
        # ]
        # if all x_i, y_i, w_i are real, remove the point
        all_sols = sols.copy()
        sols = sols[~np.all(np.isreal(sols), axis=1)]
        self.logger.info(f"Complex solutions after filtering: {sols}")

        # if there are less than 2 complex solutions, raise an error
        if len(sols) < 2:
            self.logger.error("Less than 2 complex solutions found")
            raise ValueError(
                f"Less than 2 complex solutions found! sols: {all_sols}")
        
        # Extract intersection points
        II = sols[0]
        JJ = sols[1]

        self.logger.info(f"II: {II}")
        self.logger.info(f"JJ: {JJ}")

        # Compute the dual conic of the circular points
        imDCCP = np.outer(II, JJ.T) + np.outer(JJ, II.T)
        imDCCP = imDCCP / la.norm(imDCCP)

        eigs = np.linalg.eigvals(imDCCP)

        # thresholding
        eigs[np.abs(eigs) < self.treshold] = 0

        if np.any(eigs < 0):
            self.logger.error("imDCCP is not positive definite")
            raise ValueError(
                f"imDCCP is not positive definite! No homography can be computed. eigs: {eigs}")

        self.logger.info(f"imDCCP\n: {imDCCP}")

        # Singular value decomposition
        U, S, Vt = la.svd(imDCCP)
        self.logger.info(f"U\n: {U}")
        self.logger.info(f"S\n: {S}")
        self.logger.info(f"V\n: {Vt}")

        # Compute the homography
        H = np.diag(1.0 / np.sqrt([S[0], S[1], 1.0])) @ U.T

        self.logger.info(f"H: {H}")

        return Homography(H)
