import numpy as np

from src.linear_regression import compute_cost


def test_compute_cost_initial():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    cost = compute_cost(X, y, 0, 0)

    assert np.isclose(cost, 22.0)