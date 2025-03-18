import numpy as np
from HomoTopiContinuation.Losser.FrobNormLosser import FrobNormLosser
from HomoTopiContinuation.DataStructures.datastructures import Homography


def test_FrobNormLosser_computeLossShouldReturnFrobeniusNormOfDifference():

    H1 = Homography(np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ]))

    H2 = Homography(np.array([
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 10]
    ]))

    losser = FrobNormLosser
    loss = losser.computeLoss(H1, H2)
    assert loss == np.linalg.norm(H1() - H2(), 'fro')
