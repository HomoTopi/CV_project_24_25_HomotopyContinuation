import pytest
from HomoTopiContinuation.Rectifier.numeric_rectifier import NumericRectifier
from HomoTopiContinuation.DataStructures.datastructures import Conic, Circle
import numpy as np



def test_compute_conic_line_intersection():
    # create a numeric_rectifier
    numeric_rectifier = NumericRectifier()

    # create a conic
    c1 = Circle(np.array([0, 0]), 1)
    conic = c1.to_conic()
    # create a line
    # y = - x
    line = np.array([1, 1, 1])

    intersection_points = numeric_rectifier._compute_conic_line_intersection(conic, line)

    print(intersection_points)

    assert intersection_points[0].T @ conic.M @ intersection_points[0] == 0
    assert intersection_points[1].T @ conic.M @ intersection_points[1] == 0

    print("The intersection points are on the conic")
    
    

