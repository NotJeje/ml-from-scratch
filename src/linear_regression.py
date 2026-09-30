import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

w = 0
b = 0


def compute_cost(X, y, w, b):
    m = len(X)
    y_pred = w * X + b
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


cost = compute_cost(X, y, w, b)

print(cost)