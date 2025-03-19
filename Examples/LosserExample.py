from HomoTopiContinuation.DataStructures.datastructures import Homography
from HomoTopiContinuation.ImageWarper.ImageWarper import ImageWarper
from HomoTopiContinuation.Losser.FrobNormLosser import FrobNormLosser
from HomoTopiContinuation.Losser.ReconstructionErrorLosser import ReconstructionErrorLosser
from HomoTopiContinuation.Losser.AngleDistortionLosser import AngleDistortionLosser
from HomoTopiContinuation.Losser.LinfLosser import LinfLosser
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    imWarper = ImageWarper()
    separator = '=' * 50

    def __init__(self, H_true: Homography, H_computed: Homography, points: np.ndarray, title: str):
        self.H_true = H_true
        self.H_computed = H_computed
        self.points = points
        self.title = title

    def printSeparator():
        print(Experiment.separator)

    def printLosses(self):
        Experiment.printSeparator()
        print(f'Experiment: {self.title}')
        Experiment.printSeparator()

        frobLoss = FrobNormLosser.computeLoss(self.H_true, self.H_computed)
        print(f'\tFrobenius Norm Loss: {frobLoss:.4f}')

        reconstructionErrorLoss = ReconstructionErrorLosser.computeLoss(
            self.H_true, self.H_computed, self.points)
        print(f'\tReconstruction Error Loss: {reconstructionErrorLoss:.4f}')

        angleLoss = AngleDistortionLosser.computeLoss(
            self.H_true, self.H_computed)
        print(f'\tAngle Loss: {angleLoss:.4f}')

        linfLoss = LinfLosser.computeLoss(self.H_true, self.H_computed)
        if (linfLoss is not None):
            print(f'\tLinf Loss: {linfLoss:.4f}')
        else:
            print('\tLinf Loss: Not computable')

        Experiment.printSeparator()

    def warpImage(self, img):
        return Experiment.imWarper(img, self.H_computed)


# Load the image
img = cv2.imread('./Examples/TestImages/Lena.png')

w, h = img.shape[1], img.shape[0]

points = np.array([
    [0, 0, 1],
    [w, 0, 1],
    [0, h, 1],
    [w, h, 1]
]).T

# Create a homography matrix
H_true = Homography(
    np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [1.0e-10, 0, 1]
    ])
)

# Experiment parameters

# Gausian Noise
mu = 0
sigma = 0.001

# Translation
translation_vector = np.array([10, 10, 1])

# Rotation
theta = 5 * np.pi / 180

# Skew
skew = 0.1

experiments = [
    Experiment(H_true, H_true, points, 'True Homography'),
    Experiment(H_true, Homography(
        np.array([
            [1.0, 0, translation_vector[0]],
            [0, 1.0, translation_vector[1]],
            [0.0, 0.0, translation_vector[2]]
        ])
    ) * H_true, points, 'Translation'),
    Experiment(H_true, Homography(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    ) * H_true, points, 'Rotation'),
    Experiment(H_true, Homography(
        np.array([
            [1.0, skew, 0],
            [0, 1.0, 0],
            [0, 0, 1.0]
        ])
    ) * H_true, points, 'Skew'),
    Experiment(H_true, Homography(H_true() + np.random.normal(mu, sigma, (3, 3))),
               points, 'Gaussian Noise')
]

# Plotting
plotsWidth = 3
nPlots = len(experiments) + 1
plotsHeight = int(np.ceil(nPlots / plotsWidth))
print(plotsHeight, plotsWidth)

warped_img_true = Experiment.imWarper(img, H_true)

# Display the original and warped images
plt.figure(figsize=(10, 5))
plt.subplot(plotsHeight, plotsWidth, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(plotsHeight, plotsWidth, 2)
plt.imshow(cv2.cvtColor(warped_img_true, cv2.COLOR_BGR2RGB))
plt.title('Warped Image True')
plt.axis('off')

for i, experiment in enumerate(experiments):
    experiment.printLosses()
    plt.subplot(plotsHeight, plotsWidth, i + 2)
    plt.imshow(cv2.cvtColor(experiment.warpImage(img), cv2.COLOR_BGR2RGB))
    plt.title(experiment.title)
    plt.axis('off')

plt.show()
