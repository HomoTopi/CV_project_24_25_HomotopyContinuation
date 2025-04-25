import numpy as np
from HomoTopiContinuation.Losser.CircleLosser import CircleLosser
from HomoTopiContinuation.DataStructures.datastructures import Conic, SceneDescription


def test_CircleLosser_computeEccentricityShouldReturnCorrectEccentricity():
    # Helper function to compare floating-point numbers with tolerance
    def assert_almost_equal(actual, expected, tol=1e-6):
        assert abs(
            actual - expected) <= tol, f"Expected {expected}, got {actual}"

    # Test case 1: Circle (eccentricity = 0)
    conic = Conic(np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert_almost_equal(eccentricity, 0.0)

    # Test case 2: Ellipse (eccentricity < 1)
    conic = Conic(np.array([[2, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert 0 < eccentricity < 1

    # Test case 3: Hyperbola (eccentricity > 1)
    conic = Conic(np.array([[1, 0, -1],
                            [0, -1, 0],
                            [-1, 0, 1]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert eccentricity > 1

    # Test case 4: Parabola (eccentricity = 1)
    conic = Conic(np.array([[0, 0, -1],
                            [0, 0, 0],
                            [-1, 0, 1]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert_almost_equal(eccentricity, 1.0)

    # Test case 5: Degenerate conic (eccentricity = ∞)
    conic = Conic(np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 0]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert eccentricity == float('inf')

    # Test case 6: Edge case with a nearly circular ellipse
    conic = Conic(np.array([[1.00001, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert 0 < eccentricity < 1

    # Test case 7: Conic with no definite type
    conic = Conic(np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 0]]))

    eccentricity = CircleLosser.computeEccentricity(conic)
    assert eccentricity == float('inf')

    # Test case 8: Very large coefficients
    conic = Conic(np.array([[1e6, 0, 0],
                            [0, 1e6, 0],
                            [0, 0, -1e6]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert_almost_equal(eccentricity, 0.0)
