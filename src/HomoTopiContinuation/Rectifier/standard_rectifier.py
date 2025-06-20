import numpy as np
from numpy import linalg as la
import json
from sympy import symbols, Eq, solve


from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography, Img
from HomoTopiContinuation.Rectifier.rectifier import Rectifier


class StandardRectifier(Rectifier):
    """
    A standard implementation of the Rectifier using SymPy for symbolic computation.

    This rectifier solves a system of conic equations symbolically to find intersection
    points, then computes the homography.
    """

    def computeImagesOfCircularPoints(self, C_img):
        x, y, w = symbols('x y w')
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

        eq = [Eq(a1*x**2 + b1*x*y + c1*y**2 + d1*x*w + e1*y*w + f1*w**2, 0),
              Eq(a2*x**2 + b2*x*y + c2*y**2 + d2*x*w + e2*y*w + f2*w**2, 0),
              Eq(a3*x**2 + b3*x*y + c3*y**2 + d3*x*w + e3*y*w + f3*w**2, 0)]

        # Solve the system of equations
        # TODO filter real solutions

        solutions = solve(eq, (x, y, w))
        self.logger.debug(f"Result of solve: {solutions}")

        # Extract complex solutions
        # TODO: check if casting is the same as the one used in SymPy
        sols = np.array([[complex(expr.subs({x: 1, y: 1, w: 1}))
                        for expr in expr_tuple] for expr_tuple in solutions])
        self.logger.info(f"Solutions before filtering: {sols}")

        if len(sols) == 0:
            self.logger.error("No solutions found")
            raise ValueError(
                f"No solutions found! sols: {sols}")

        return sols

    def rectify(self, C_img: Conics, returnCP: bool = False) -> Homography:
        """
        Rectify a pair of conics using SymPy.

        Returns:
            Homography: The rectification homography
        """

        self.logger.info("Rectifying using scipy")

        self.logger.info(f"Conics: {C_img}")

        sols = self.computeImagesOfCircularPoints(C_img)

        imDCCP = self.compute_imDCCP_from_solutions(sols)
        # TODO: compute the rectification homography both by svd and by a fully homotopy continuation approach
        H = self._compute_h_from_svd(imDCCP)

        if returnCP:
            return H, sols
        return H


if __name__ == "__main__":
    # Create a StandardRectifier instance
    rectifier = StandardRectifier()

    # Call the rectify method to test it
    rectifier.rectify()
