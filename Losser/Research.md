# How to compute how good a shape reconstructing homography is?

## Introduction
In this project we need to compare different image rectification algorithms, it thus become imperative to have a metric to compare the quality of the rectification. In this document we will discuss how to compute a metric to compare the quality of the rectification.

## Problem Statement
Given an image that is correctly rectified by applying the homography $H_{true}$, an algorithm computes a homography $H_{computed}$ to rectify the image. We need to compute a metric to compare the quality of the rectification.

## Interestig Observations
> The Frobinious norm is invariant to rotations. This is because it is the sum of the squares of the singular values of the matrix. Since rotations do not change the singular values of a matrix, the frobinious norm is invariant to rotations.

## Idea #1 - Compute the frobinious norm of the difference between the two homographies
The first idea that comes to mind is to compute the frobinious norm of the difference between the two homographies. The frobinious norm of the difference between the two homographies is given by:
$$
FN(\text{Frobinious Norm}) = \sqrt{\sum_{i=1}^{3} \sum_{j=1}^{3} (H_{true}[i,j] - H_{computed}[i,j])^2}
$$

Since we are working in homogenous coordinates, the matrices must be normalized before computing the frobinious norm. The normalized homography is given by:
$$
H_{normalized} = \frac{H}{H[3,3]}
$$

### Effecto of random noise
Let's consider the effect of random Guassian noise on the frobinious norm of the difference between the two homographies. For this we can consider $H_{computed} = H_{true} + \varepsilon$ where $\varepsilon$ is a 3 by 3 matrix of random i.i.d. Guassian random variables: $\varepsilon_{i,j} \sim \mathcal{N}(0, \sigma^2)$. The indipendence of the entries of $\varepsilon$ is a strong assumption, but it is a good starting point.

Let's now compute the frobinious norm of the difference between the two homographies. 
$$
FN_{\varepsilon} = \sqrt{\sum_{i=1}^{3} \sum_{j=1}^{3} (H_{true}[i,j] - (H_{true}[i,j] + \varepsilon[i,j]))^2} = \sqrt{\sum_{i=1}^{3} \sum_{j=1}^{3} \varepsilon[i,j]^2}
$$

Thanks to the i.i.d. assumption, the problem becomes a sum of the squares of 9 i.i.d. Guassian random variables. 
$$
FN_{\varepsilon} = \sqrt{\sum_{i=1}^{9} X_i^2} \text{  where  } X_i \sim \mathcal{N}(0, \sigma^2) 
$$
Equivalently, we can express it in terms of normal random variables:
$$
FN_{\varepsilon} = \sqrt{\sum_{i=1}^{9} (\sigma Z_i) ^2} = \sigma \sqrt{\sum_{i=1}^{9} (Z_i) ^2} \text{  where  } Z_i \sim \mathcal{N}(0, 1)
$$

The sum of the squares of 9 i.i.d. Guassian random variables is a Chi-squared distribution with 9 degrees of freedom. The frobinious norm of the difference between the two homographies is thus a Chi-squared distribution with 9 degrees of freedom scaled by $\sigma$.
$$
FN_{\varepsilon} \sim \sigma \sqrt{\chi^2(9)}
$$

Thus:

$$
\mathbb{E}[FN_{\varepsilon}] = 3 \sigma \\
\text{Var}[FN_{\varepsilon}] = 3\sqrt{2} \sigma^2 \text{ To be checked}
$$
## Idea #2 - Compare the images of the line at infinity
The second idea is to compare the lines that are moved to infinity by the homographies. The line at infinity is given by the last row of the homography matrix. We can compute the line at infinity for both the true and computed homographies and compare them.
In particular, we can compute the angle between the two lines at infinity. The angle between two lines is given by:
$$
\text{Angle} = \cos^{-1} \left( \frac{L_{true} \cdot L_{computed}}{\|L_{true}\| \|L_{computed}\|} \right)
$$

## Idea #3 - Study the "error Homography"
The third idea is to compute the homography that maps the image rectified by the computed homography to the image rectified by the true homography. This homography is called the "error homography". We can then compute the frobinious norm of the error homography to compare the quality of the rectification.
$$
H_{error} = H_{true}^{-1} H_{computed}
$$

The closer this homography is to the identity matrix, the better the rectification.

Since we are doing image reconstruction we are agnostic to similarity transformations, we thus need a metric on how far is the error homography from a similiraity transformation. See [this paper](./Papers/FAST%20AND%20INTERPRETABLE%202D%20HOMOGRAPHY%20DECOMPOSITION.pdf) for the SKS decomposition of the homography.
$$
H_{error} = H_{S2}^{-1} H_{K} H_{S1}
$$
Where $H_{S1}$ and $H_{S2}$ are similarity transformations and $H_{K}$ is a homography with 4 degrees of freedom. We can then compute the frobinious norm of $H_{K}$ to compare the quality of the rectification.

## Idea #4 - Compute the reprojection error
The fourth idea is to compute the reprojection error. We can compute the reprojection error by projecting the corners of the image rectified by the true homography to the image rectified by the computed homography. We can then compute the distance between the projected corners and the actual corners.
Given the corners of the image to be rectified $C = \{[0, 0, 1]^T, [w, 0, 1]^T, [0, h, 1]^T, [w, h, 1]^T\}$ where $w$ and $h$ are the width and height of the image respectively. The corners of the image rectified by the true homography are given by $C_{true} = H_{true} C$. The corners of the image rectified by the computed homography are given by $C_{computed} = H_{computed} C$. The reprojection error is given by:
$$
RE = \sum_{i=1}^{4} \|C_{true}[i] - C_{computed}[i]\| = \sum_{i=1}^{4} \|H_{true} C[i] - H_{computed} C[i]\| = \\
\sum_{i=1}^{4} \|(H_{true} - H_{computed}) C[i]\|
$$
