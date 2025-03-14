import numpy as np
from numpy import linalg as la


class SceneDescription:
    """
    Description of a scene with two circles and camera parameters.
    
    Attributes:
        f (float): Focal length
        theta (float): Camera angle
        circle1 (list): Parameters of the first circle [centerX, centerY, radius]
        circle2 (list): Parameters of the second circle [centerX, centerY, radius]
    """
    
    def __init__(self, f, theta, circle1, circle2):
        """
        Initialize a SceneDescription object.
        
        Args:
            f (float): Focal length
            theta (float): Camera angle
            circle1 (list): Parameters of the first circle [centerX, centerY, radius]
            circle2 (list): Parameters of the second circle [centerX, centerY, radius]
            
        Raises:
            ValueError: If any circle has a negative radius
        """
        if circle1[2] < 0 or circle2[2] < 0:
            raise ValueError("Circle radius must be positive")
        
        self.f = f
        self.theta = theta
        self.circle1 = circle1
        self.circle2 = circle2


class Conic:
    """
    Representation of a conic section using a 3x3 symmetric matrix.
    
    Attributes:
        M (numpy.ndarray): A 3x3 symmetric matrix representing the conic
    """
    
    def __init__(self, M):
        """
        Initialize a Conic object.
        
        Args:
            M (numpy.ndarray): A 3x3 symmetric matrix
            
        Raises:
            ValueError: If M is not a 3x3 matrix or not symmetric
        """
        if M.shape != (3, 3):
            raise ValueError(f"Conic matrix must be 3×3, got {M.shape}")
        
        if not np.allclose(M, M.T):
            raise ValueError("Conic matrix must be symmetric")
        
        self.M = M
    
    def to_algebraic_form(self):
        """
        Convert from matrix form to algebraic parameters form.
        
        Returns:
            tuple: The algebraic coefficients (a, b, c, d, e, f) where
                  the conic equation is ax² + bxy + cy² + dx + ey + f = 0
        """
        return (
            self.M[0, 0],         # a
            self.M[0, 1] * 2,     # b
            self.M[1, 1],         # c
            self.M[0, 2] * 2,     # d
            self.M[1, 2] * 2,     # e
            self.M[2, 2]          # f
        )


class Conics:
    """
    A pair of conics.
    
    Attributes:
        C1 (Conic): First conic
        C2 (Conic): Second conic
    """
    
    def __init__(self, C1, C2):
        """
        Initialize a Conics object with two conics.
        
        Args:
            C1 (Conic): First conic
            C2 (Conic): Second conic
        """
        self.C1 = C1
        self.C2 = C2


class Homography:
    """
    A homography transformation represented by a matrix.
    
    Attributes:
        H (numpy.ndarray): The homography matrix
    """
    
    def __init__(self, H):
        """
        Initialize a Homography object.
        
        Args:
            H (numpy.ndarray): The homography matrix
        """
        self.H = H


class Img:
    """
    An image with a true homography and conics.
    
    Attributes:
        h_true (Homography): The true homography
        C_img (Conics): The pair of conics in the image
    """
    
    def __init__(self, h_true, C_img):
        """
        Initialize an Img object.
        
        Args:
            h_true (Homography): The true homography
            C_img (Conics): The pair of conics in the image
        """
        self.h_true = h_true
        self.C_img = C_img 