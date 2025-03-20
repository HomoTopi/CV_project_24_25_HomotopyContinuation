import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CameraPlotter:
    def __init__(self, center=np.array([0.0, 0.0, 0.0]), yaw=0, pitch=0, roll=0, size=1.0, color='blue', bodyRatio=0.5):
        self.center = center
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.size = size
        self.color = color
        self.bodyRatio = bodyRatio

    def rotationMatrix(self, yaw, pitch, roll):
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
        self.plotCube(ax)
        self.plot_pyramid(ax, base_size=self.size, height=self.size)
