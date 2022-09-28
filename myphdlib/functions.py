import numpy as np

def relu(x, m, x0, lb):
    """
    Rectified linear unit activation function
    """

    y = m * (x - x0) + lb
    mask = x < x0
    y[mask] = np.full(mask.sum(), lb)

    return y

def sigmoid(x, a, b, c, d):
    """
    Adaptive sigmoid function
    """

    return a / (1.0 + np.exp(-c * (x - d))) + b

def exponential(x, a, h, k):
    """
    """

    return a * np.exp(x - h) + k

from scipy.optimize import leastsq

def f(x, t, a1, b, c, d, a2, h, k):
    mask = np.arange(x.size) < t
    result = np.full(x.size, np.nan)
    result[mask] = sigmoid(x[mask], a1, b, c, d)
    result[~mask] = exponential(x[~mask], a2, h, k)
    return result

def loss(params, x, y):
    result = f(x, *params)
    residual = y - result
    return residual

class SigmoidalExponentialModel():

    def fit(self, x, y):
        """
        """

        p0 = np.ones(8)
        p0[0] = int(x.size / 2) + 2
        result = leastsq(loss, p0, (x, y))
        self.params = result[0]

        return

    def predict(self, x):
        """
        """
        
        return f(x, *self.params)