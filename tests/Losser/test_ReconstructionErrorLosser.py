import numpy as np
from HomoTopiContinuation.Losser.ReconstructionErrorLosser import ReconstructionErrorLosser
from HomoTopiContinuation.DataStructures.datastructures import Homography


def test_ReconstructionErrorLosser_computeLossShouldBeZeroForPerfectReconstruction():
    H1 = Homography(np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 10]
    ]))

    # random 3 by 4 matrix
    points = np.random.rand(3, 4)

    losser = ReconstructionErrorLosser
    loss = losser.computeLoss(H1, H1, points)
    assert np.abs(loss) < 1e-6
