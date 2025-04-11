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
    # create a linealpha * (lamb **3) + beta * (lamb **2) * mu + gamma * lamb * (mu ** 2) + delta * (mu ** 3)
    # y = - x + 1
    line = np.array([1, 1, -1])

    intersection_points = numeric_rectifier._compute_conic_line_intersection(conic, line)

    assert abs(intersection_points[0].T @ conic.M @ intersection_points[0]) < 1e-3
    assert abs(intersection_points[1].T @ conic.M @ intersection_points[1]) < 1e-3


def test_compute_conics_intersection():
    # create a numeric_rectifier
    numeric_rectifier = NumericRectifier()

    # create two conics
    c1 = Circle(np.array([0, 0]), 1)
    c2 = Circle(np.array([1, 1]), 5)
    conic1 = c1.to_conic()
    conic2 = c2.to_conic()

    result = numeric_rectifier._compute_conics_intersection(conic1, conic2)
    print("result: ", result)
    assert result.size > 0
    for pt in result:
        print(pt.T @ conic1.M @ pt)
        print(pt.T @ conic2.M @ pt)
        assert abs(pt.T @ conic1.M @ pt) < 1e-3
        assert abs(pt.T @ conic2.M @ pt) < 1e-3

def test_cubic_equation_solver():
    numeric_rectifier = NumericRectifier()
    c1 = Circle(np.array([0, 0]), 1)
    c2 = Circle(np.array([1, 1]), 2)
    conic1 = c1.to_conic()
    conic1.M[0,0] = 2
    conic2 = c2.to_conic()
    
    
    
    alpha, beta, gamma, delta = numeric_rectifier._get_parameters(conic1, conic2)
    #lamb, mu = numeric_rectifier._compute_cubic_equation(alpha, beta,gamma,delta)
    lamb, mu = numeric_rectifier._solve_symbolic_cubic_equation(alpha, beta,gamma,delta)
    lamb = lamb[0]
    mu = mu[0]
    res = alpha * (lamb **3) + beta * (lamb **2) * mu + gamma * lamb * (mu ** 2) + delta * (mu ** 3)
    assert abs(res) < 10e-3