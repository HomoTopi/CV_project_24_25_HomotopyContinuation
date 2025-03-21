import numpy as np
import matplotlib.pyplot as plt
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic, Circle, SceneDescription
from HomoTopiContinuation.Plotter.CameraPlotter import CameraPlotter
import HomoTopiContinuation.SceneGenerator.scene_generator as sg
import seaborn as sns


class Plotter:
    """
    A class for plotting both 2D and 3D plots.
    There are methods to plot conics and cameras.

    To use the plotter, create a Plotter object specifying the number of plots in x and y directions and the figure size.
    For every plot first create a new axis using newAxis or new3DAxis methods.
    The newly created axis will be set as the current axis for subsequent plots.
    Then use the plotConic2D, plotConic3D, and plotCamera methods to plot conics and cameras.
    """

    def __init__(self, nPlotsx=1, nPlotsy=1, figsize=(10, 10), title="My plot"):
        """
        Initialize a Plotter object.

        Args:
            nPlotsx (int): Number of plots in x direction
            nPlotsy (int): Number of plots in y direction
            figsize (tuple): Figure size
            title (str): Title of the figure
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

    def getCurrentAxis(self) -> plt.Axes:
        """
        Get the current axis.

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The current axis
        """
        return self.ax

    def newAxis(self, title="", axisSame=True):
        """
        Create a new 2D axis. The new axis will be set as the current axis for subsequent plots.

        Args:
            title (str): Title of the axis
            axisSame (bool): If True, the axis will have the same scale

        Raises:
            ValueError: If the maximum number of plots is reached

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
        Create a new 3D axis. The new axis will be set as the current axis for subsequent plots.

        Args:
            title (str): Title of the axis
            axisSame (bool): If True, the axis will have the same scale

        Raises:
            ValueError: If the maximum number of plots is reached
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
        """
        Plot a 2D conic.

        Args:
            conic (Conic): The conic to plot
            x_range (tuple): The range of x values expressed as (start, end, number of points)
            y_range (tuple): The range of y values expressed as (start, end, number of points)
            conicName (str): The name of the conic
            color (str): The color of the conic

        Raises:
            ValueError: If the current axis is not 2D
        """

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

        Args:
            center (numpy.ndarray): The center of the camera expressed as a 3D point
            yaw (float): The yaw angle of the camera in degrees
            pitch (float): The pitch angle of the camera in degrees
            roll (float): The roll angle of the camera in degrees
            size (float): The size of the camera
            color (str): The color of the camera
            bodyRatio (float): The ratio of the body the ratio between the body height and the body width, default 0.5
        """
        if (self.dimention != 3):
            raise ValueError("The current axis is not 3D.")
        camera = CameraPlotter(center, yaw, pitch, roll,
                               size, color, bodyRatio)
        camera.plotCamera(self.ax)

    def plotConic3D(self, conic: Conic, sceneDescription: sg.SceneDescription, x_range=(-1, 1, 100), y_range=(-1, 1, 100), conicName='Conic', color='r', tol=1e-2):
        """
        Plot a 3D conic.

        Args:
            conic (Conic): The conic to plot
            sceneDescription (sg.SceneDescription): The description of the scene
            x_range (tuple, optional): The range of x values expressed as (start, end, number of points). Defaults to (-1, 1, 100).
            y_range (tuple, optional): The range of y values expressed as (start, end, number of points). Defaults to (-1, 1, 100).
            conicName (str, optional): The name of the conic. Defaults to 'Conic'.
            color (str, optional): The color of the plot. Defaults to 'r'.
            tol (_type_, optional): The tolerance for the plot
            mask. Points with Z values less than tol will be plotted. Defaults to 1e-2.

        Raises:
            ValueError: If the current axis is not 3D
        """
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
        Shows the plot.
        """
        plt.show()


if __name__ == "__main__":
    # Initialize the plotter
    plotter = Plotter(title="Conics", nPlotsx=2, nPlotsy=1)

    # Create a conic
    conic = Conic(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ]))

    # First plot (on the left)
    plotter.newAxis(title="Conic")
    plotter.plotConic2D(conic)

    # Second plot (on the right)
    c = Circle(np.array([0, 0]), 1)
    conic = c.to_conic()
    sd = sg.SceneDescription(1, 30, np.array([0, 0, 1]), c, c)
    plotter.new3DAxis(title="Conic")
    plotter.plotCamera()
    plotter.plotConic3D(conic, sd)

    # Show the plot
    plotter.show()
