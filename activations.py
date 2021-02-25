import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2

class Tanh:
    def __init__(self):
        pass

    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)*np.tanh(x)