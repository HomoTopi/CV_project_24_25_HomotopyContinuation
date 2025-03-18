from HomoTopiContinuation.DataStructures.datastructures import Homography
from HomoTopiContinuation.ImageWarper.ImageWarper import ImageWarper
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('./Examples/TestImages/Lena.png')
assert img is not None

# Create a homography matrix
H = Homography(
    np.array([
        [1.0, 0.2, 0.3],
        [0.2, 1.0, 0.1],
        [0, 0, 1.0]
    ])
)

# Warp the image
warped_img = ImageWarper()(img, H)

# Display the original and warped images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
plt.title('Warped Image')
plt.axis('off')

plt.show()
