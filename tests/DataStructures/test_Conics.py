from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic
import numpy as np
import pytest


def test_Conic_constructorShouldThrowExceptionForNonSymmetricMatrices():
    with pytest.raises(ValueError):
        C = Conic(np.array([
            [1.0, 0.2],
            [0.3, 1]
        ]))
