from HomoTopiContinuation.DataStructures.datastructures import Homography
from HomoTopiContinuation.ImageWarper.ImageWarper import ImageWarper
from HomoTopiContinuation.Losser.FrobNormLosser import FrobNormLosser
from HomoTopiContinuation.Losser.ReconstructionErrorLosser import ReconstructionErrorLosser
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('./Examples/TestImages/Lena.png')

# Create a homography matrix
H_true = Homography(
    np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0]
    ])
)

mu = 0
sigma = 0.001
H_computed = Homography(
    H_true() + np.random.normal(mu, sigma, H_true().shape)
)

w, h = img.shape[1], img.shape[0]

points = np.array([
    [0, 0, 1],
    [w, 0, 1],
    [0, h, 1],
    [w, h, 1]
]).T

# Compute the losses
frobLoss = FrobNormLosser.computeLoss(H_true, H_computed)
print(f'Frobenius Norm Loss: {frobLoss}')

reconstructionErrorLoss = ReconstructionErrorLosser.computeLoss(
    H_true, H_computed, points)
print(f'Reconstruction Error Loss: {reconstructionErrorLoss}')


# Warp the image
warped_img_true = ImageWarper()(img, H_true)
warped_img_computed = ImageWarper()(img, H_computed)

# Display the original and warped images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(warped_img_true, cv2.COLOR_BGR2RGB))
plt.title('Warped Image True')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(warped_img_computed, cv2.COLOR_BGR2RGB))
plt.title('Warped Image Computed')
plt.axis('off')

plt.show()
