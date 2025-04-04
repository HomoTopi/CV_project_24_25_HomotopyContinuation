import pytest
from HomoTopiContinuation.Rectifier.numeric_rectifier import NumericRectifier
from HomoTopiContinuation.DataStructures.datastructures import Conic
import numpy as np



def test_compute_conic_line_intersection():
    # create a numeric_rectifier
    numeric_rectifier = NumericRectifier()

    # create a conic
    conic = Conic(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    # create a line
    line = np.array([1, 0.5, 1])

    # compute the intersection points
    intersection_points = numeric_rectifier._compute_conic_line_intersection(conic, line)

    # assert the intersection points are not None
    assert intersection_points is not None
    
    # assert the intersection points are a numpy array
    assert isinstance(intersection_points, np.ndarray)

    
    

