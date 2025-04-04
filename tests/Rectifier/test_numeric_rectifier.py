import pytest
from HomoTopiContinuation.Rectifier.numeric_rectifier import NumericRectifier
from HomoTopiContinuation.DataStructures.datastructures import Conic, Circle
import numpy as np



def test_compute_conic_line_intersection():
    # create a numeric_rectifier
    numeric_rectifier = NumericRectifier()

    # create a conic
    c1 = Circle(np.array([0, 0]), 2)
    conic = c1.to_conic()
    # create a line
    # y = - 10x - 100
    line = np.array([1, 10, 100])

    intersection_points = numeric_rectifier._compute_conic_line_intersection(conic, line)

    assert abs(intersection_points[0].T @ conic.M @ intersection_points[0]) < 1e-5
    assert abs(intersection_points[1].T @ conic.M @ intersection_points[1]) < 1e-5


def test_compute_conics_intersection():
    # create a numeric_rectifier
    numeric_rectifier = NumericRectifier()

    # create two conics
    c1 = Circle(np.array([0, 0]), 2)
    c2 = Circle(np.array([1, 1]), 3)
    conic1 = c1.to_conic()
    conic2 = c2.to_conic()

    numeric_rectifier._compute_conics_intersection(conic1, conic2)
    
    assert False

