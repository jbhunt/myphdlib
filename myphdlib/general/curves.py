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
    Basic exponential function
    """

    return a * np.exp(x - h) + k