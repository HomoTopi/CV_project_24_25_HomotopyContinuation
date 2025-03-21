import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CameraPlotter:
    """
    A class for plotting cameras in 3D.
    """

    def __init__(self, center=np.array([0.0, 0.0, 0.0]), yaw=0, pitch=0, roll=0, size=1.0, color='blue', bodyRatio=0.5):
        """
        Initialize a CameraPlotter object.

        Args:
            center (np.ndarray, optional): The center of the camera. Defaults to np.array([0.0, 0.0, 0.0]).
            yaw (int, optional): The yaw angle of the camera in degrees. Defaults to 0.
            pitch (int, optional): The pitch angle of the camera in degrees. Defaults to 0.
            roll (int, optional): The roll angle of the camera in degrees. Defaults to 0.
            size (float, optional): The size of the camera. It is the biggest dimention of the camera body. Defaults to 1.0.
            color (str, optional): The color of the camera plot. Defaults to 'blue'.
            bodyRatio (float, optional): The ratio between the body width and the body height. Defaults to 0.5.
        """
        self.center = center
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.size = size
        self.color = color
        self.bodyRatio = bodyRatio

    def rotationMatrix(self, yaw: int, pitch: int, roll: int) -> np.ndarray:
        """
        Compute the rotation matrix from yaw, pitch, and roll angles.

        Args:
            yaw (_type_): The yaw angle in degrees
            pitch (_type_): The pitch angle in degrees
            roll (_type_): The roll angle in degrees

        Returns:
            np.ndarray: The rotation matrix
        """
        yaw, pitch, roll = np.radians([yaw, pitch, roll])

        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        return R_yaw @ R_pitch @ R_roll

    def plotCube(self, ax: plt.Axes):
        """
        Plot a cube.

        Args:
            ax (plt.Axes): The axis to plot the cube on
        """
        halfSize = self.size / 2
        vertices = np.array([
            [-halfSize*self.bodyRatio, -halfSize*self.bodyRatio, -halfSize],
            [halfSize*self.bodyRatio, -halfSize*self.bodyRatio, -halfSize],
            [halfSize*self.bodyRatio, halfSize*self.bodyRatio, -halfSize],
            [-halfSize*self.bodyRatio, halfSize*self.bodyRatio, -halfSize],
            [-halfSize*self.bodyRatio, -halfSize*self.bodyRatio, halfSize],
            [halfSize*self.bodyRatio, -halfSize*self.bodyRatio, halfSize],
            [halfSize*self.bodyRatio, halfSize*self.bodyRatio, halfSize],
            [-halfSize*self.bodyRatio, halfSize*self.bodyRatio, halfSize]
        ])

        R = self.rotationMatrix(self.yaw, self.pitch, self.roll)
        rotatedVertices = (R @ vertices.T).T

        translatedVertices = rotatedVertices + self.center

        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5]
        ]

        for face in faces:
            faceVertices = np.array([translatedVertices[i] for i in face])
            ax.add_collection3d(Poly3DCollection(
                [faceVertices], facecolors=self.color, linewidths=1, edgecolors='k', alpha=0.5))

    def plot_pyramid(self, ax: plt.Axes, base_size=1.0, height=1.0):
        """
        Plot a pyramid.

        Args:
            ax (plt.Axes): The axis to plot the pyramid on
            base_size (float, optional): The size of the base of the pyramid. Defaults to 1.0.
            height (float, optional): The height of the pyramid. Defaults to 1.0.
        """
        # Define pyramid vertices (apex at origin)
        half_base = base_size / 2
        vertices = np.array([
            [-half_base, -half_base, height],  # Bottom-left corner
            [half_base, -half_base, height],   # Bottom-right corner
            [half_base, half_base, height],    # Top-right corner
            [-half_base, half_base, height],   # Top-left corner
            [0, 0, 0]                # Apex
        ])

        # Apply rotation
        R = self.rotationMatrix(self.yaw, self.pitch, self.roll)
        rotated_vertices = (R @ vertices.T).T

        # Translate to the center position
        translated_vertices = rotated_vertices + self.center

        # Define the pyramid faces
        faces = [
            [0, 1, 4],  # Front face
            [1, 2, 4],  # Right face
            [2, 3, 4],  # Back face
            [3, 0, 4],  # Left face
            [0, 1, 2, 3]  # Base face
        ]

        # Plot the faces
        for face in faces[:-1]:  # Plot triangular faces
            triangle = [translated_vertices[vert] for vert in face]
            ax.add_collection3d(Poly3DCollection(
                [triangle], alpha=0.5, color=self.color, linewidths=1, edgecolors='k'))

        # Plot the base
        base = [translated_vertices[vert] for vert in faces[-1]]
        ax.add_collection3d(Poly3DCollection(
            [base], alpha=0.5, color=self.color, linewidths=1, edgecolors='k'))

    def plotCamera(self, ax: plt.Axes):
        """
        Plot the camera.

        Args:
            ax (plt.Axes): The axis to plot the camera on
        """
        self.plotCube(ax)
        self.plot_pyramid(ax, base_size=self.size, height=self.size)
