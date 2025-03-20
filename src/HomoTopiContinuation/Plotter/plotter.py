import numpy as np
import matplotlib.pyplot as plt
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic, Circle, SceneDescription
from HomoTopiContinuation.Plotter.CameraPlotter import CameraPlotter
import HomoTopiContinuation.SceneGenerator.scene_generator as sg
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
            # Set the axis to have the same scale
            self.ax.set_aspect('equal', adjustable='datalim')

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

    def plotCamera(self, center=np.array([0.0, 0.0, 0.0]), yaw=0, pitch=0, roll=0, size=.2, color='blue', bodyRatio=0.5):
        """
        Plot a camera.
        """
        if (self.dimention != 3):
            raise ValueError("The current axis is not 3D.")
        camera = CameraPlotter(center, yaw, pitch, roll,
                               size, color, bodyRatio)
        camera.plotCamera(self.ax)

    def plotConic3D(self, conic: Conic, sceneDescription: sg.SceneDescription, x_range=(-1, 1, 100), y_range=(-1, 1, 100), conicName='Conic', color='r', tol=1e-2):
        if (self.dimention != 3):
            raise ValueError("The current axis is not 3D.")

        a, b, c, d, e, f = conic.to_algebraic_form()

        x = np.linspace(x_range[0], x_range[1], x_range[2])
        y = np.linspace(y_range[0], y_range[1], y_range[2])
        X, Y = np.meshgrid(x, y)

        Z = a * X**2 + b * X * Y + c * Y**2 + d * X + e * Y + f

        mask = np.abs(Z) < tol
        X = X[mask]
        Y = Y[mask]

        referenceMatrix = sg.SceneGenerator.compute_reference_matrix(
            sceneDescription)

        points = np.array([X, Y, np.ones_like(X)])
        points = referenceMatrix @ points

        self.ax.scatter(points[0], points[1], points[2],
                        label=conicName, color=color)

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

    c = Circle(np.array([0, 0]), 1)
    conic = c.to_conic()

    sd = sg.SceneDescription(1, 30, np.array([0, 0, 1]), c, c)

    plotter.new3DAxis(title="Conic")
    plotter.plotCamera()
    plotter.plotConic3D(conic, sd)
    plotter.show()
