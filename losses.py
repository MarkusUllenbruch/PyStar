import numpy as np

class MSE:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def derivative(self, y_true, y_pred):
        return y_pred - y_true