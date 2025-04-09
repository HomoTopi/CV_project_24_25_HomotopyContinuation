import numpy as np
from HomoTopiContinuation.Losser.CircleLosser import CircleLosser
from HomoTopiContinuation.DataStructures.datastructures import Conic, SceneDescription


def test_CircleLosser_computeEccentricityShouldReturnCorrectEccentricity():
    # Test case 1: Circle (eccentricity = 0)
    conic = Conic(np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert eccentricity == 0.0

    # Test case 2: Ellipse (eccentricity < 1)
    conic = Conic(np.array([[1, 0, -1],
                            [0, 4, -4],
                            [-1, -4, 8]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert 0 < eccentricity < 1

    # Test case 3: Hyperbola (eccentricity > 1)
    conic = Conic(np.array([[1, 0, -1],
                            [0, -4, 4],
                            [-1, 4, -8]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert eccentricity > 1

    # Test case 4: Parabola (eccentricity = 1)
    conic = Conic(np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert eccentricity == 1.0

    # Test case 5: Degenerate conic (eccentricity = ∞)
    conic = Conic(np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert eccentricity == float('inf')
