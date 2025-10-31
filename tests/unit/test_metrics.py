import numpy as np
from rice2025.metrics import euclidean, manhattan

def test_euclidean_right_triangle():
    assert euclidean(np.array([0,0]), np.array([3,4])) == 5.0

def test_manhattan_distance():
    assert manhattan([1,2], [4,6]) == 7.0
