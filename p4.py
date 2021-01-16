import numpy as np


def main():
    X = np.array([[2, 9], [1, 5], [3, 6]])
    y = np.array(([30], [12], [60]))

    X = X / np.max(X, axis=0)
    y = y / 100

    nn = TwoLayerNN()

    yp, l = None, None
    for i in range(10000):
        yp, l = nn.train(X, y, 5e-1)

    print(
        "avg loss:",
        np.average(np.abs(l)) * 100,
        "predicted:",
        yp[:, 0] * 100,
        "actual:",
        y[:, 0] * 100,
    )


class TwoLayerNN:
    def __init__(self):
        self.insize = 2
        self.outsize = 1
        self.hiddensize = 3

        self.W1 = np.random.uniform(size=(self.insize, self.hiddensize))
        self.W2 = np.random.uniform(size=(self.hiddensize, self.outsize))

    def train(self, X, y, lr):
        # forward
        h1 = X @ self.W1
        h2 = sigmoid(h1)
        h3 = h2 @ self.W2
        yp = sigmoid(h3)
        l = y - yp

        # backward
        dYp = l * d_sigmoid(yp)
        dW2 = h2 @ dYp
        dW1 = X.T @ (dYp @ self.W2.T * d_sigmoid(h2))

        # gradient descent
        self.W2 += dW2 * lr
        self.W1 += dW1 * lr

        return yp, l


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(s):
    return s * (1 - s)


if __name__ == "__main__":
    main()
