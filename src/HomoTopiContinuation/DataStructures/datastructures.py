import numpy as np
from numpy import linalg as la


class Conic:
    """
    Representation of a conic section using a 3x3 symmetric matrix.

    Attributes:
        M (numpy.ndarray): A 3x3 symmetric matrix representing the conic
    """

    def __init__(self, M: np.ndarray):
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
                  a b/2 d/2
                  b/2 c e/2
                  d/2 e/2 f
        """
        return (
            self.M[0, 0],         # a
            self.M[0, 1] * 2,     # b
            self.M[1, 1],         # c
            self.M[0, 2] * 2,     # d
            self.M[1, 2] * 2,     # e
            self.M[2, 2]          # f
        )

    def __call__(self):
        """
        Return the conic matrix.

        Returns:
            numpy.ndarray: The conic matrix
        """
        return self.M

    def applyHomography(self, H: 'Homography') -> 'Conic':
        """
        Apply a homography to the circle.

        Args:
            H (Homography): The homography

        Returns:
            Conic: The conic after applying the homography
        """
        return Conic(H.inv().T @ self.M @ H.inv())


class Conics:
    """
    A pair of conics.

    Attributes:
        C1 (Conic): First conic
        C2 (Conic): Second conic
        C3 (Conic): Third conic
    """

    def __init__(self, C1: Conic, C2: Conic, C3: Conic):
        """
        Initialize a Conics object with two conics.

        Args:
            C1 (Conic): First conic
            C2 (Conic): Second conic
            C3 (Conic): Third conic
        """
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

    def __call__(self):
        """
        Return the pair of conics.

        Returns:
            tuple: The pair of conics (C1, C2, C3)
        """
        return self.C1, self.C2, self.C3
    
    def __str__(self):
        """
        Return a string representation of the pair of conics.
        """
        return f"""C1:\n{self.C1.M}\nC2:\n{self.C2.M}\nC3:\n{self.C3.M}"""


class Circle:
    def __init__(self, center: np.ndarray, radius: float):
        """
        Initialize a Circle object.

        Args:
            center (numpy.ndarray): The center of the circle
            radius (float): The radius of the circle
        Raises:
            ValueError: If the radius is negative
        """
        if radius < 0:
            raise ValueError("Circle radius must be positive")

        self.center = center
        self.radius = radius

    def to_conic(self):
        """
        Convert the circle to a conic.
        """
        conic = np.array([
            [1, 0, -self.center[0]],
            [0, 1, -self.center[1]],
            [-self.center[0], -self.center[1], self.center[0]
                ** 2 + self.center[1]**2 - self.radius**2]
        ])
        return Conic(conic)


class SceneDescription:
    """
    Description of a scene with two circles and camera parameters.

    Attributes:
        f (float): Focal length
        y_rotation (float): Camera angle in degrees around the y-axis
        offset (numpy.ndarray): Offset of the camera from the origin
        circle1 (circle): First circle
        circle2 (circle): Second circle
        cicrle3 (circle): Third circle
    """

    def __init__(self, f: float, y_rotation: float, offset: np.ndarray, circle1: Circle, circle2: Circle, circle3: Circle):
        """
        Initialize a SceneDescription object.

        Args:
            f (float): Focal length
            y_rotation (float): Rotation angle around the y-axis in degrees 
            offset (numpy.ndarray): Offset of the camera from the origin
            circle1 (Circle): Parameters of the first circle
            circle2 (Circle): Parameters of the second circle
            circle3 (Circle): Parameters of the third circle

        """
        if f <= 0:
            raise ValueError("Focal length must be positive")
        if offset.shape != (3,):
            raise ValueError("Offset must be a 3D vector")
        self.f = f
        self.y_rotation = y_rotation
        self.circle1 = circle1
        self.circle2 = circle2
        self.circle3 = circle3
        self.offset = offset


class Homography:
    """
    A homography transformation represented by a matrix.

    Attributes:
        H (numpy.ndarray): The homography matrix, it should be a 3x3 invertible matrix
    """

    def __init__(self, H: np.ndarray):
        """
        Initialize a Homography object.

        Args:
            H (numpy.ndarray): The homography matrix
        """
        if (H.shape != (3, 3)):
            raise ValueError(f"Homography matrix must be 3×3, got {H.shape}")
        if np.abs(la.det(H)) < 1e-6:
            raise ValueError(
                "Homography matrix must be invertible, det(H) = " + str(la.det(H)))

        self.H = H

    def __call__(self) -> np.ndarray:
        """
        Return the homography matrix.

        Returns:
            numpy.ndarray: The homography matrix
        """
        return self.H

    def inv(self) -> np.ndarray:
        """
        Return the inverse of the homography matrix.

        Returns:
            numpy.ndarray: The inverse of the homography matrix
        """
        return la.inv(self.H)

    def Hinv(self) -> 'Homography':
        """
        Return the inverse of the homography matrix.

        Returns:
            Homography: The inverse of the homography matrix
        """
        return Homography(self.inv())

    def __mul__(self, other: 'Homography') -> 'Homography':
        """
        Multiply two homographies.

        Args:
            other (Homography): The other homography to multiply with

        Returns:
            Homography: The result of the multiplication
        """
        return Homography(self.H @ other.H)


class Img:
    """
    An image with a true homography and conics.

    Attributes:
        h_true (Homography): The true homography
        C_img (Conics): The pair of conics in the image
    """

    def __init__(self, h_true: Homography, C_img: Conics):
        """
        Initialize an Img object.

        Args:
            h_true (Homography): The true homography
            C_img (Conics): The pair of conics in the image
        """
        self.h_true = h_true
        self.C_img = C_img
