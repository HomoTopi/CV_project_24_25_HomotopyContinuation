import numpy as np
import matplotlib.pyplot as plt
from HomoTopiContinuation.DataStructures.datastructures import Conic, Circle, Img, Homography, Conics
from HomoTopiContinuation.Plotter.CameraPlotter import CameraPlotter
import HomoTopiContinuation.SceneGenerator.scene_generator as sg
import seaborn as sns


class Plotter:
    """
    A class for plotting both 2D and 3D plots.
    It contains methods to plot conics and cameras.

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
        self.dimension = 2

    def getCurrentAxis(self) -> plt.Axes:
        """
        Get the current axis.

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The current axis
        """
        return self.ax

    def newAxis(self, title="", axisSame=True, legend=True):
        """
        Create a new 2D axis. The new axis will be set as the current axis for subsequent plots.

        Args:
            title (str): Title of the axis
            axisSame (bool): If True, the axis will have the same scale
            legend (bool): If True, the legend will be shown

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
        self.dimension = 2

    def drawLegend(self):
        """
        Draw the legend.
        """
        self.ax.legend()

    def new3DAxis(self, title=""):
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

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.dimension = 3

    def scaleAxis3D(self):
        """
        Scale the 3D axis to have the same scale.

        Raises:
            ValueError: If the current axis is not 3D
        """
        if (self.dimension != 3):
            raise ValueError("The current axis is not 3D.")
        limits = np.array(
            [self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        self.ax.set_xlim3d([center[0] - radius, center[0] + radius])
        self.ax.set_ylim3d([center[1] - radius, center[1] + radius])
        self.ax.set_zlim3d([center[2] - radius, center[2] + radius])

    def plotConic2D(self, conic: Conic, x_NPoints=100, y_NPoints=100, conicName='Conic', color='r', alpha=1.0):
        """
        Plot a 2D conic.

        Args:
            conic (Conic): The conic to plot
            x_NPoints (int): The number of points to plot the conic in x direction, default 100
            y_NPoints (int): The number of points to plot the conic in y direction, default 100
            conicName (str): The name of the conic
            color (str): The color of the conic

        Raises:
            ValueError: If the current axis is not 2D
        """

        if (self.dimension != 2):
            raise ValueError("The current axis is not 2D.")
        a, b, c, d, e, f = conic.to_algebraic_form()

        (x_min, y_min), (x_max, y_max) = conic.compute_bounding_box()

        x = np.linspace(x_min, x_max, x_NPoints)
        y = np.linspace(y_min, y_max, y_NPoints)
        X, Y = np.meshgrid(x, y)

        Z = a * X**2 + b * X * Y + c * Y**2 + d * X + e * Y + f

        cs = self.ax.contour(X, Y, Z, levels=[0], colors=color, alpha=alpha)
        plt.clabel(cs, inline=True, fontsize=10, fmt=conicName)

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

        Raises:
            ValueError: If the current axis is not 3D
        """
        if (self.dimension != 3):
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
            tol (float, optional): The tolerance for the plot
            mask. Points with Z values less than tol will be plotted. Defaults to 1e-2.

        Raises:
            ValueError: If the current axis is not 3D
        """
        if (self.dimension != 3):
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

    def plotCircle2D(self, circle: Circle, thetaResolution=100, color='r', name='Circle', alpha=1.0):
        """
        Plot a 2D circle.

        Args:
            circle (Circle): The circle to plot
            thetaResolution (int): The number of points to plot the circle
            color (str): The color of the circle
            name (str): The name of the circle

        Raises:
            ValueError: If the current axis is not 2D
        """
        if (self.dimension != 2):
            raise ValueError("The current axis is not 2D.")
        theta = np.linspace(0, 2 * np.pi, thetaResolution)
        x = circle.center[0] + circle.radius * np.cos(theta)
        y = circle.center[1] + circle.radius * np.sin(theta)

        x = np.append(x, x[0])
        y = np.append(y, y[0])

        self.ax.plot(x, y, color=color, label=name, alpha=alpha)

    def plotCircle3D(self, circle: Circle, sceneDescription: sg.SceneDescription, thetaResolution=100, color='r', name='Circle'):
        """
        Plot a 3D circle.

        Args:
            circle (Circle): The circle to plot
            sceneDescription (sg.SceneDescription): The description of the scene
            thetaResolution (int): The number of points to plot the circle
            color (str): The color of the circle
            name (str): The name of the circle

        Raises:
            ValueError: If the current axis is not 3D
        """
        if (self.dimension != 3):
            raise ValueError("The current axis is not 3D.")
        theta = np.linspace(0, 2 * np.pi, thetaResolution)
        x = circle.center[0] + circle.radius * np.cos(theta)
        y = circle.center[1] + circle.radius * np.sin(theta)

        referenceMatrix = sg.SceneGenerator.compute_reference_matrix(
            sceneDescription)

        points = np.array([x, y, np.ones_like(x)])
        points = referenceMatrix @ points

        self.ax.plot(points[0], points[1], points[2], color=color, label=name)

    def drawReferenceFrame(self, sceneDescription: sg.SceneDescription, size=1, colorX='r', colorY='g'):
        """
        Draw the reference frame of the scene described by sceneDescription. It will draw the X and Y axis of the reference plane.

        Args:
            sceneDescription (sg.SceneDescription): The description of the scene
            size (float): The size of the reference frame
            colorX (str): The color of the X axis
            colorY (str): The color of the Y axis

        Raises:
            ValueError: If the current axis is not 3D
        """
        if (self.dimension != 3):
            raise ValueError("The current axis is not 3D.")
        referenceMatrix = sg.SceneGenerator.compute_reference_matrix(
            sceneDescription)

        i_hat = referenceMatrix[:, 0]
        j_hat = referenceMatrix[:, 1]
        offset = referenceMatrix[:, 2]

        self.ax.quiver(offset[0], offset[1], offset[2], i_hat[0], i_hat[1], i_hat[2],
                       color=colorX, label='X', length=size, arrow_length_ratio=0.1)

        self.ax.quiver(offset[0], offset[1], offset[2], j_hat[0], j_hat[1], j_hat[2],
                       color=colorY, label='Y', length=size, arrow_length_ratio=0.1)

    def plotSceneNewPlotter(sceneDescription: sg.SceneDescription, img: Img, colorC1='r', colorC2='g', colorC3='b', name='Scene') -> 'Plotter':
        """
        Plot a scene with the original image, the rectified image, and the 3D scene.
        This will produce three plots:
        - A rectified plot with the circles
        - A 3D plot with the camera and the circles
        - A plot of the rendered image with the conics

        This method creates a new plotter object and plots the scene.

        Args:
            sceneDescription (sg.SceneDescription): The description of the scene
            img (Img): The image of the scene
            colorC1 (str, optional): The color for the plotting of the first conic. Defaults to 'r'.
            colorC2 (str, optional): The color for the plotting of the second conic. Defaults to 'g'.
            colorC3 (str, optional): The color for the plotting of the third conic. Defaults to 'b'.
            name (str, optional): The name of the scene. Defaults to 'Scene'.

        Returns:
            Plotter: The Plotter object
        """
        plotter = Plotter(nPlotsx=3, nPlotsy=1, figsize=(10, 10), title=name)

        plotter.plotScene(sceneDescription, img, colorC1,
                          colorC2, colorC3, name)

        plotter.show()

        return plotter

    def plotScene(self, sceneDescription: sg.SceneDescription, img: Img, colorC1='r', colorC2='g', colorC3='b', name='Scene'):
        """
        Plot a scene with the original image, the rectified image, and the 3D scene.
        This will produce three plots:
        - A rectified plot with the circles
        - A 3D plot with the camera and the circles
        - A plot of the rendered image with the conics

        This requires at least 3 plots remaining.

        Args:
            sceneDescription (sg.SceneDescription): The description of the scene
            img (Img): The image of the scene
            colorC1 (str, optional): The color for the plotting of the first conic. Defaults to 'r'.
            colorC2 (str, optional): The color for the plotting of the second conic. Defaults to 'g'.
            colorC3 (str, optional): The color for the plotting of the third conic. Defaults to 'b'.
            name (str, optional): The name of the scene. Defaults to 'Scene'.

        Raises:
            ValueError: If the maximum number of plots is reached
        """

        if (self.plotNumber + 2 > self.maxPlots):
            raise ValueError(
                "Maximum number of plots reached. Please create a new figure.")

        alpha = 0.2 if (sceneDescription.noiseScale > 0) else 1.0

        # First plot (on the Left)
        self.newAxis(title="Original Image", axisSame=True)
        self.plotConic2D(
            img.C_img.C1, conicName="Conic 1 Original", color=colorC1, alpha=alpha)
        self.plotConic2D(
            img.C_img.C2, conicName="Conic 2 Original", color=colorC2, alpha=alpha)
        self.plotConic2D(
            img.C_img.C3, conicName="Conic 3 Original", color=colorC3, alpha=alpha)

        if (sceneDescription.noiseScale > 0):
            self.plotConic2D(
                img.C_img_noise.C1, conicName="Conic 1 Noisy", color=colorC1)
            self.plotConic2D(
                img.C_img_noise.C2, conicName="Conic 2 Noisy", color=colorC2)
            self.plotConic2D(
                img.C_img_noise.C3, conicName="Conic 3 Noisy", color=colorC3)
        self.drawLegend()

        # Second plot (in the middle)
        self.new3DAxis(title="3D Scene")
        self.plotCamera()  # keep default body ratio
        self.drawReferenceFrame(sceneDescription)
        self.plotCircle3D(sceneDescription.circle1,
                          sceneDescription, color=colorC1, name="Circle 1")
        self.plotCircle3D(sceneDescription.circle2,
                          sceneDescription, color=colorC2, name="Circle 2")
        self.plotCircle3D(sceneDescription.circle3,
                          sceneDescription, color=colorC3, name="Circle 3")
        self.drawLegend()
        self.scaleAxis3D()

        # Third plot (on the right)
        self.newAxis(title="Rectified Image", axisSame=True)
        self.plotCircle2D(sceneDescription.circle1,
                          name="Circle 1", color=colorC1, alpha=alpha)
        self.plotCircle2D(sceneDescription.circle2,
                          name="Circle 2", color=colorC2, alpha=alpha)
        self.plotCircle2D(sceneDescription.circle3,
                          name="Circle 3", color=colorC3, alpha=alpha)

        if (sceneDescription.noiseScale > 0):
            h_true_inv = Homography(img.h_true.inv())
            C1_best = img.C_img_noise.C1.applyHomography(
                h_true_inv)
            C2_best = img.C_img_noise.C2.applyHomography(
                h_true_inv)
            C3_best = img.C_img_noise.C3.applyHomography(
                h_true_inv)
            self.plotConic2D(
                C1_best, conicName="Circle 1 Best", color=colorC1)
            self.plotConic2D(
                C2_best, conicName="Circle 2 Best", color=colorC2)
            self.plotConic2D(
                C3_best, conicName="Circle 3 Best", color=colorC3)
        self.drawLegend()

    def plotExperiment(self, sceneDescription: sg.SceneDescription, img: Img, warpedConics: Conics, colorC1='r', colorC2='g', colorC3='b', name='Scene', warpedConicsDrawingNPoints=100):
        """
        Plot the experiment with the original image, the rectified image, and the 3D scene.
        This will produce four plots:
        - A rectified plot with the circles
        - A 3D plot with the camera and the circles
        - A plot of the rendered image with the conics
        - A plot of the warped conics
        This requires at least 4 plots remaining.

        Args:
            sceneDescription (sg.SceneDescription): The description of the scene
            img (Img): The rendered image of the scene
            warpedConics (Conics): The recrtified conics
            colorC1 (str, optional): The color for the plotting of the first conic. Defaults to 'r'.
            colorC2 (str, optional): The color for the plotting of the second conic. Defaults to 'g'.
            colorC3 (str, optional): The color for the plotting of the third conic. Defaults to 'b'.
            name (str, optional): The name of the scene. Defaults to 'Scene'.
            warpedConicsDrawingNPoints (int, optional): The number of points used draw the conics. Defaults to 100.
        """
        if (self.maxPlots < 4):
            raise ValueError(
                "You need at least 4 plots to plot the experiment.")
        if (self.plotNumber + 3 > self.maxPlots):
            raise ValueError(
                "You need at least 4 plots to plot the experiment. Please create a new figure.")

        self.plotScene(sceneDescription, img, colorC1,
                       colorC2, colorC3, name)

        self.newAxis("Reconstructed Rectification", axisSame=True)

        self.plotConic2D(
            warpedConics.C1, conicName="C1", color="red", x_NPoints=warpedConicsDrawingNPoints, y_NPoints=warpedConicsDrawingNPoints)
        self.plotConic2D(
            warpedConics.C2, conicName="C2", color="green", x_NPoints=warpedConicsDrawingNPoints, y_NPoints=warpedConicsDrawingNPoints)
        self.plotConic2D(
            warpedConics.C3, conicName="C3", color="blue", x_NPoints=warpedConicsDrawingNPoints, y_NPoints=warpedConicsDrawingNPoints)

    def show(self):
        """
        Shows the plot.
        """
        plt.show()


if __name__ == "__main__":
    # Define the circles
    c1 = Circle(np.array([0, 0]), 1)
    c2 = Circle(np.array([0.5, 0]), 1)
    c3 = Circle(np.array([0, 0.5]), 1)

    # Define the scene description
    sd = sg.SceneDescription(1, 30, np.array([0, 0, 2]), c1, c2, c3)

    # Generate Image
    image = sg.SceneGenerator.generate_scene(sd)

    Plotter.plotSceneNewPlotter(
        sd, image, name='Scene')
