import numpy as np
import logging
from ..DataStructures.datastructures import Conic, Conics
from .standard_rectifier import StandardRectifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """
    Example demonstrating the usage of the rectifier library.
    """
    # Create two conic matrices
    # These are just example conics - replace with real data for actual use
    M1 = np.array([
        [1.0, 0.2, 0.3],
        [0.2, 1.0, 0.1],
        [0.3, 0.1, -1.0]
    ])

    M2 = np.array([
        [1.2, -0.1, 0.4],
        [-0.1, 0.8, 0.2],
        [0.4, 0.2, -1.5]
    ])

    # Create Conic objects
    C1 = Conic(M1)
    C2 = Conic(M2)

    # Create a Conics object containing both conics
    conics = Conics(C1, C2)

    # Create the rectifier
    rectifier = StandardRectifier()

    # Perform rectification
    try:
        homography = rectifier.rectify(conics)
        print("Rectification successful!")
        print("Homography matrix:")
        print(homography.H)
    except Exception as e:
        print(f"Rectification failed: {e}")


if __name__ == "__main__":
    main()
