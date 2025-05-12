import numpy as np
from numpy import linalg as la


class DistortionParams:
    """
    Parameters for the distortion of the conic.
    """
    def __init__(self, k1: float, k2: float, p1: float, p2: float, k3: float):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3

class Conic:
    """
    Representation of a conic section using a 3x3 symmetric matrix.
    A conic with algebraic form given by ax² + bxy + cy² + dx + ey + f = 0
    is represented as the matrix

    a b/2 d/2
    b/2 c e/2
    d/2 e/2 f

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

        self._M = M

    @property
    def M(self) -> np.ndarray:
        """
        Return the conic matrix.

        Returns:
            float: The conic matrix
        """
        return self._M

    @M.setter
    def M(self, value):
        """
        Set the conic matrix.
        """
        self._M = value

    def to_algebraic_form(self) -> tuple:
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

    def __call__(self) -> np.ndarray:
        """
        Return the conic matrix.

        Returns:
            numpy.ndarray: The conic matrix
        """
        return self.M

    def applyHomography(self, H: 'Homography') -> 'Conic':
        """
        Apply a homography to the conic.

        Args:
            H (Homography): The homography

        Returns:
            Conic: The conic after applying the homography
        """
        return Conic(H.inv().T @ self.M @ H.inv())

    def get_non_zero_entry(self) -> tuple[int, int]:
        """
        Get the first non-zero entry of the conic matrix.
        """
        return np.nonzero(self.M)[0][0], np.nonzero(self.M)[1][0]

    def randomize(self, noiseScale: float = 0.1) -> 'Conic':
        """
        Randomize the conic matrix by adding Gaussian noise.

        Args:
            noiseScale (float): The scale of the noise to be added

        Returns:
            Conic: The randomized conic
        """
        noise = np.random.normal(0, noiseScale, self.M.shape)

        # make the noise symmetric
        noise = noise + noise.T - np.diag(np.diag(noise))
        noise = noise / 2

        return Conic(self.M + noise)

    @staticmethod
    def fit_conic(points: np.ndarray) -> 'Conic':
        """
        Fit a conic to a set of points applying a least squares method.
        
        Args:
            points (numpy.ndarray): Array of shape (N, 2) containing N points (x,y)
            
        Returns:
            Conic: The fitted conic
        """
        A = np.zeros((points.shape[0], 6))
        for i in range(points.shape[0]):
            A[i] = np.array([points[i, 0]**2, points[i, 0] * points[i, 1], points[i, 1]**2, points[i, 0], points[i, 1], 1])
        
        # Apply constraint f=1 (rightmost coefficient is 1)
        # This solves the homogeneous system Ax=0 with the constraint
        if points.shape[0] > 5:  # Need at least 5 points for a unique solution
            # Normalize last column to 1
            A_prime = A[:, :5]
            b = -A[:, 5]
            params = np.linalg.lstsq(A_prime, b, rcond=None)[0]
            coef = np.append(params, 1.0)  # [a, b, c, d, e, f=1]
        else:
            raise ValueError("At least 6 points are required to fit a conic")
        
        # Convert algebraic form to matrix form
        # [a b/2 d/2]
        # [b/2 c e/2]
        # [d/2 e/2 f]
        M = np.array([
            [coef[0], coef[1]/2, coef[3]/2],
            [coef[1]/2, coef[2], coef[4]/2],
            [coef[3]/2, coef[4]/2, coef[5]]
        ])
        
        return Conic(M)

    def is_ellipse(self) -> bool:
        """
        Check if the conic is an ellipse.
        A conic is an ellipse if the determinant of the top-left 2x2 submatrix is positive.
        """
        A = self.M[:2, :2]
        det_A = np.linalg.det(A)
        return det_A > 0

    def is_parabola(self) -> bool:
        """
        Check if the conic is a parabola.
        A conic is a parabola if the determinant of the top-left 2x2 submatrix is zero.
        """
        A = self.M[:2, :2]
        det_A = np.linalg.det(A)
        return np.isclose(det_A, 0)

    def center(self) -> np.ndarray:
        """
        Compute the center of the ellipse represented by the conic matrix.

        Returns:
            numpy.ndarray: The center of the ellipse
        """
        if self.is_parabola():
            raise ValueError("Conic has no center because it is a parabola")

        A = self.M[:2, :2]
        b = self.M[:2, 2]
        c = self.M[2, 2]

        center = -np.linalg.inv(A) @ b
        return center

    def compute_bounding_box(self) -> tuple:
        """
        Compute the bounding box of the ellipse represented by the conic matrix.

        If the conic is not an ellipse, return ((0, 0), (0, 0)).

        Returns:
            tuple: The bounding box coordinates (min_x, min_y), (max_x, max_y)
        """
        if not self.is_ellipse():
            return ((-100, -100), (100, 100))

        M = self.M/self.M[2, 2]  # Normalize the conic matrix
        # Compute the center of the ellipse
        A = self.M[:2, :2]
        b = self.M[:2, 2]
        c = self.M[2, 2]

        # Center of the ellipse
        center = self.center()
        cen_x, cen_y = center[0], center[1]

        # Center the conic matrix
        M = self.centered().M

        # Compute semi-minor axes using the eigenvalues of the conic matrix
        eigenvalues, _ = np.linalg.eig(M[:2, :2])
        semi_axes = np.sqrt(1 / np.abs(eigenvalues))
        semi_axes = np.abs(semi_axes)
        semi_axes = np.real(semi_axes)
        max_axis = np.max(semi_axes)

        min_x, max_x = cen_x - max_axis, cen_x + max_axis
        min_y, max_y = cen_y - max_axis, cen_y + max_axis

        return ((min_x, min_y), (max_x, max_y))

    def centered(self) -> 'Conic':
        """
        Center the conic matrix.

        Returns:
            Conic: The centered conic matrix
        """
        center = self.center()
        A = self.M[:2, :2]
        b = self.M[:2, 2]
        c = self.M[2, 2]

        # Center the conic matrix
        k = c + b.T @ center + center.T @ b + center.T @ A @ center
        M_centered = self.M.copy() / -k
        return Conic(M_centered)


class Conics:
    """
    A data structure containing three conics.

    Attributes:
        C1 (Conic): First conic
        C2 (Conic): Second conic
        C3 (Conic): Third conic
    """

    def __init__(self, C1: Conic, C2: Conic, C3: Conic):
        """
        Initialize a Conics object with three conics.

        Args:
            C1 (Conic): First conic
            C2 (Conic): Second conic
            C3 (Conic): Third conic
        """
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

    def __call__(self) -> tuple:
        """
        Return the three conics.

        Returns:
            tuple: The three conics (C1, C2, C3)
        """
        return self.C1, self.C2, self.C3

    def __str__(self) -> str:
        """
        Return a string representation of the conics.

        Returns:
            str: String representation of the conics
        """
        return f"""C1:\n{self.C1.M}\nC2:\n{self.C2.M}\nC3:\n{self.C3.M}"""

    def __iter__(self):
        """
        Return an iterator over the conics.
        """
        return iter([self.C1, self.C2, self.C3])

    def to_json(self):
        """
        Convert the Conics object to a JSON serializable format.
        """
        return {
            "C1": self.C1.M.tolist(),
            "C2": self.C2.M.tolist(),
            "C3": self.C3.M.tolist()
        }

    def from_json(json_str):
        """
        Create a Conics object from a JSON string.
        """
        json_str['C1'] = np.array(json_str['C1'])
        json_str['C2'] = np.array(json_str['C2'])
        json_str['C3'] = np.array(json_str['C3'])
        return Conics(Conic(json_str['C1']), Conic(json_str['C2']), Conic(json_str['C3']))


class Circle:
    def __init__(self, center: np.ndarray, radius: float):
        """
        Initialize a Circle object.

        Args:
            center (numpy.ndarray): The center of the circle
            radius (float): The radius of the circle
        Raises:
            ValueError: If the radius is not positive
        """
        if radius <= 0:
            raise ValueError("Circle radius must be positive")

        self.center = center
        self.radius = radius

    def to_conic(self) -> 'Conic':
        """
        Convert the circle to a conic.

        Returns:
            Conic: the representation of the circle as a conic
        """
        conic = np.array([
            [1, 0, -self.center[0]],
            [0, 1, -self.center[1]],
            [-self.center[0], -self.center[1], self.center[0]
                ** 2 + self.center[1]**2 - self.radius**2]
        ])

        return Conic(conic)

    def sample_points(self, n: int) -> np.ndarray:
        """
        Sample n points on the circle.
        """
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        z = np.ones(n)
        return np.vstack((x, y, z)).T  # shape (N, 3)


class SceneDescription:
    """
    Description of a scene with three circles and camera parameters.

    Attributes:
        f (float): Focal length of the camera
        y_rotation (float): Camera rotation angle in degrees around the y-axis
        offset (numpy.ndarray): Offset of the camera from the origin
        circle1 (circle): First circle
        circle2 (circle): Second circle
        circle3 (circle): Third circle
    """

    def __init__(self, f: float, y_rotation: float, offset: np.ndarray, circle1: Circle, circle2: Circle, circle3: Circle, noiseScale: float = 0):
        """
        Initialize a SceneDescription object.

        Args:
            f (float): Focal length of the camera
            y_rotation (float): Camera rotation angle around the y-axis in degrees 
            offset (numpy.ndarray): Offset of the camera from the origin
            circle1 (Circle): Parameters of the first circle
            circle2 (Circle): Parameters of the second circle
            circle3 (Circle): Parameters of the third circle
            noiseScale (float): Scale of the noise to be added to the circles

        Raises:
            ValueError: If the focal length is not a positive number, or if the offset vector does not have the expected shape.
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
        self.noiseScale = noiseScale

    def from_json(json_str) -> 'SceneDescription':
        """
        Create a SceneDescription object from a JSON string.
        """
        json_str['offset'] = np.array(json_str['offset'])
        json_str['circle1'] = Circle(
            np.array(json_str['circle1']['center']), json_str['circle1']['radius'])
        json_str['circle2'] = Circle(
            np.array(json_str['circle2']['center']), json_str['circle2']['radius'])
        json_str['circle3'] = Circle(
            np.array(json_str['circle3']['center']), json_str['circle3']['radius'])
        json_str['f'] = float(json_str['f'])
        json_str['y_rotation'] = float(json_str['y_rotation'])
        json_str['noiseScale'] = float(json_str['noiseScale'])

        return SceneDescription(
            json_str['f'],
            json_str['y_rotation'],
            json_str['offset'],
            json_str['circle1'],
            json_str['circle2'],
            json_str['circle3'],
            json_str['noiseScale']
        )

    def to_json(self) -> dict:
        """
        Convert the SceneDescription object to a JSON serializable format.
        """
        return {
            "f": self.f,
            "y_rotation": self.y_rotation,
            "offset": self.offset.tolist(),
            "circle1": {
                "center": self.circle1.center.tolist(),
                "radius": self.circle1.radius
            },
            "circle2": {
                "center": self.circle2.center.tolist(),
                "radius": self.circle2.radius
            },
            "circle3": {
                "center": self.circle3.center.tolist(),
                "radius": self.circle3.radius
            }
        }


class Homography:
    """
    A homography transformation represented by a matrix.

    Attributes:
        H (numpy.ndarray): The homography matrix, it must be a 3x3 invertible matrix
    """

    def __init__(self, H: np.ndarray, threshold: float = 1e-6):
        """
        Initialize a Homography object.
        Set to 0 all the elements of H with a magnitude less than a threshold and
        set to 0 all the real or imaginary parts of H with a magnitude less than threshold.

        Args:
            H (numpy.ndarray): The homography matrix

        Raises:
            ValueError: If the homography matrix is not 3x3 or not invertible
        """
        self.threshold = threshold
        if (H.shape != (3, 3)):
            raise ValueError(f"Homography matrix must be 3×3, got {H.shape}")
        if np.abs(la.det(H)) < self.threshold:
            raise ValueError(
                "Homography matrix must be invertible, det(H) = " + str(la.det(H)))

        # set to 0 all the elements of H with a magnitude less than threshold
        H[np.abs(H) < self.threshold] = 0

        # set to 0 all the real or imaginary parts of H with a magnitude less than threshold
        H = np.real_if_close(H, tol=self.threshold)
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

    def to_json(self):
        """
        Convert the Homography object to a JSON serializable format.
        """
        return {
            "H": self.H.tolist()
        }

    def from_json(json_str) -> 'Homography':
        """
        Create a Homography object from a JSON string.
        """
        json_str['H'] = np.array(json_str['H'])
        return Homography(json_str['H'])


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

    def to_json(self):
        """"
        Convert the Img object to a JSON serializable format.
        """
        return {
            "h_true": self.h_true.H.tolist(),
            "C_img": {
                "C1": self.C_img.C1.M.tolist(),
                "C2": self.C_img.C2.M.tolist(),
                "C3": self.C_img.C3.M.tolist()
            }
        }

    def from_json(json_str):
        """
        Create an Img object from a JSON string.
        """
        json_str['h_true'] = np.array(json_str['h_true'])
        json_str['C_img']['C1'] = np.array(json_str['C_img']['C1'])
        json_str['C_img']['C2'] = np.array(json_str['C_img']['C2'])
        json_str['C_img']['C3'] = np.array(json_str['C_img']['C3'])
        return Img(Homography(json_str['h_true']), Conics(Conic(json_str['C_img']['C1']), Conic(json_str['C_img']['C2']), Conic(json_str['C_img']['C3'])))
