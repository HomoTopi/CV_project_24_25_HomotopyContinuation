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
    conic = Conic(np.array([[-2.10813559,  0.17186159,  7.6791946,],
                            [0.17186159,  0.01631471,  0.10557325,],
                            [7.6791946,   0.10557325, - 1.40891357,]]))
    eccentricity = CircleLosser.computeEccentricity(conic)
    assert eccentricity > 1

    # Test case 4: Parabola (eccentricity = 1)
    conic = Conic(np.array([[-1, 0, 0],
                            [0, 0, 0.5],
                            [0, 0.5, 0]]))
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


def test_CircleLosser_eccentricityInvariantTranslations():
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

    # Test case 2: Circle with translation (eccentricity = 0)
    conic_translated = Conic(np.array([[1, 0, 2],
                                       [0, 1, 3],
                                       [2, 3, -1]]))
    eccentricity_translated = CircleLosser.computeEccentricity(
        conic_translated)
    assert_almost_equal(eccentricity_translated, 0.0)

    # Test case 3: Ellipse (eccentricity < 1)
    conic_ellipse = Conic(np.array([[2, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, -1]]))
    eccentricity_ellipse = CircleLosser.computeEccentricity(conic_ellipse)
    assert 0 < eccentricity_ellipse < 1

    # Test case 4: Ellipse with translation (eccentricity < 1)
    conic_ellipse_translated = Conic(np.array([[2, 0, 2],
                                               [0, 1, 3],
                                               [2, 3, -1]]))
    eccentricity_ellipse_translated = CircleLosser.computeEccentricity(
        conic_ellipse_translated)
    assert 0 < eccentricity_ellipse_translated < 1

    assert np.isclose(eccentricity_ellipse, eccentricity_ellipse_translated,
                      atol=1e-6), "Eccentricity should be invariant under translation"
