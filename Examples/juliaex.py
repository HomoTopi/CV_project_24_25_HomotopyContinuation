
try:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    pass

import numpy as np

from HomoTopiContinuation.Rectifier.homotopyc_rectifier import HomotopyContinuationRectifier
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic

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

rectifier = HomotopyContinuationRectifier()
rectifier.rectify(conics)