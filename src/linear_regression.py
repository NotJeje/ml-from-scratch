import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])


def compute_cost(X, y, w, b):
    m = len(X)
    y_pred = w * X + b
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost


def gradient_descent(X, y, w, b, learning_rate, iterations):
    m = len(X)
    cost_history = []

    for i in range(iterations):
        y_pred = w * X + b

        dw = (1 / m) * np.sum((y_pred - y) * X)
        db = (1 / m) * np.sum(y_pred - y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

    return w, b, cost_history


def main():
    w = 0
    b = 0

    learning_rate = 0.01
    iterations = 1000

    w, b, cost_history = gradient_descent(
        X, y, w, b, learning_rate, iterations
    )

    print("w:", w)
    print("b:", b)
    print("First cost:", cost_history[0])
    print("Last cost:", cost_history[-1])

    plt.plot(cost_history, color="purple", linewidth=2)

    plt.scatter(0, cost_history[0], color="red", label="Start")
    plt.scatter(
        len(cost_history) - 1,
        cost_history[-1],
        color="green",
        label="End"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Convergence")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()