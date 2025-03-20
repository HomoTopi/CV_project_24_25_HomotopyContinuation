import numpy as np
import matplotlib.pyplot as plt
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic, Circle

class Plotter:

    def plot_conics(self, conics: Conics, title: str):
        """
        Plot the conics.

        Args:
            conics (Conics): The pair of conics
        """
        C1 = conics.C1
        C2 = conics.C2
        # Extract coefficients of conics
        a1, b1, c1, d1, e1, f1 = C1.to_algebraic_form()
        a2, b2, c2, d2, e2, f2 = C2.to_algebraic_form()
        # Range of x and y values
        x = np.linspace(-2, 2, 1000)
        y = np.linspace(-2, 2, 1000)
        X, Y = np.meshgrid(x, y)
        # Compute Z values for both conics
        Z1 = a1*X**2 + b1*X*Y + c1*Y**2 + d1*X + e1*Y + f1
        Z2 = a2*X**2 + b2*X*Y + c2*Y**2 + d2*X + e2*Y + f2
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.contour3D(X, Y, Z1, levels=[0], colors='r')
        ax.contour3D(X, Y, Z2, levels=[0], colors='b')
        #ax.contour3D(X, Y, Z1, colors='r')
        #ax.contour3D(X, Y, Z2, colors='b')
        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    def plot_conic(self, conic: Conic, title: str):
        """
        Plot the conic.

        Args:
            conic (Conic): The conic
        """
        # Extract coefficients of conic
        a, b, c, d, e, f = conic.to_algebraic_form()
        # Range of x and y values
        x = np.linspace(-500, 500, 1000)
        y = np.linspace(-500, 500, 1000)
        X, Y = np.meshgrid(x, y)
        # Compute Z values for conic
        Z = a*X**2 + b*X*Y + c*Y**2 + d*X + e*Y + f
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.contour3D(X, Y, Z, levels=[0], colors='r')
        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1])
        plt.show()