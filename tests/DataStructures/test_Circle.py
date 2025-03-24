from HomoTopiContinuation.DataStructures.datastructures import Circle
import numpy as np
import pytest


def test_Circle_toconicShouldConstructTheRightConicMatrix():
    C = Circle(np.array([1, 2]), 3)
    conic = C.to_conic()
    for x in np.linspace(-10, 10, 100):
        for y in np.linspace(-10, 10, 100):
            point = np.array([x, y, 1]).T
            if (x - 1) ** 2 + (y - 2) ** 2 == 3 ** 2:
                assert point.T @ conic.M @ point == 0
            else:
                assert point.T @ conic.M @ point != 0

def test_Circle_constructorShouldThrowExceptionForNegativeRadius():
    with pytest.raises(ValueError):
        Circle(np.array([1, 2]), -3)
    