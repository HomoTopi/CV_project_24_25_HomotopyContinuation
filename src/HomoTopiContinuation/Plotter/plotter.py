import numpy as np
import matplotlib.pyplot as plt
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic, Circle
from HomoTopiContinuation.Plotter.CameraPlotter import CameraPlotter
import seaborn as sns


class Plotter:
    def __init__(self, nPlotsx=1, nPlotsy=1, figsize=(10, 10), title="My plot"):
        """
        Initialize a Plotter object.
        """
        sns.set_theme()
        self.figure = plt.figure(figsize=figsize)
        self.figure.suptitle(title)
        self.ax = None
        self.plotNumber = 1
        self.maxPlots = nPlotsx * nPlotsy
        self.nPlotsx = nPlotsx
        self.nPlotsy = nPlotsy
        self.dimention = 2

    def newAxis(self, title="", axisSame=True):
        """
        Create a new axis.
        """
        if (self.plotNumber > self.maxPlots):
            raise ValueError(
                "Maximum number of plots reached. Please create a new figure.")

        self.ax = self.figure.add_subplot(
            self.nPlotsy, self.nPlotsx, self.plotNumber)
        self.plotNumber += 1

        self.ax.set_title(title)
        self.ax.set_aspect('equal', adjustable='datalim')
        if axisSame:
            self.ax.axis('equal')
        self.dimention = 2

    def new3DAxis(self, title="", axisSame=True):
        """
        Create a new 3D axis.
        """
        if (self.plotNumber > self.maxPlots):
            raise ValueError(
                "Maximum number of plots reached. Please create a new figure.")
        self.ax = self.figure.add_subplot(
            self.nPlotsy, self.nPlotsx, self.plotNumber,
            projection='3d')
        self.ax.set_title(title)
        self.plotNumber += 1
        if axisSame:
            self.ax.set_box_aspect([1, 1, 1])

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.dimention = 3

    def plotConic2D(self, conic: Conic, x_range=(-1, 1, 100), y_range=(-1, 1, 100), conicName='Conic', color='r'):
        if (self.dimention != 2):
            raise ValueError("The current axis is not 2D.")
        a, b, c, d, e, f = conic.to_algebraic_form()

        x = np.linspace(x_range[0], x_range[1], x_range[2])
        y = np.linspace(y_range[0], y_range[1], y_range[2])
        X, Y = np.meshgrid(x, y)

        Z = a * X**2 + b * X * Y + c * Y**2 + d * X + e * Y + f

        self.ax.contour(X, Y, Z, levels=[0], colors=color, label=conicName)

    def plotCamera(self, center=np.array([0.0, 0.0, 0.0]), yaw=0, pitch=0, roll=0, size=.1, color='blue', bodyRatio=0.5):
        """
        Plot a camera.
        """
        if (self.dimention != 3):
            raise ValueError("The current axis is not 3D.")
        camera = CameraPlotter(center, yaw, pitch, roll,
                               size, color, bodyRatio)
        camera.plotCamera(self.ax)

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
        # ax.contour3D(X, Y, Z1, colors='r')
        # ax.contour3D(X, Y, Z2, colors='b')
        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    def show(self):
        """
        Show the plot.
        """
        plt.show()


if __name__ == "__main__":
    plotter = Plotter(title="Conics", nPlotsx=2, nPlotsy=1)
    conic = Conic(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ]))
    plotter.newAxis(title="Conic")
    plotter.plotConic2D(conic)

    plotter.new3DAxis(title="Conic")
    plotter.plotCamera()
    plotter.show()
