import numpy as np

# Base class o a Layer
class Layer:
    def __init__(self):
        self.input = None
        self.z = None
        self.a = None

    # computes output Y of a layer for given input X
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_neurons, output_neurons, activation):
        super().__init__()
        self.activation = activation
        self.weights = np.random.rand(output_neurons, input_neurons) - 0.5
        self.bias = np.random.rand(output_neurons, 1) - 0.5

    def forward(self, inputs):

        self.input = inputs
        self.z = np.dot(self.weights, self.input)
        self.z = self.z + np.tile(self.bias, (1, self.z.shape[1]))
        self.a = self.activation(self.z)
        return self.a

    def backward(self, output_error):
        '''Backpropagation/ Gradients Calculation'''
        delta = output_error * self.activation.derivative(self.z)
        self.grad_w = np.dot(delta, self.input.transpose()) / delta.shape[1]
        self.grad_b = np.sum(delta, axis=1, keepdims=True) / delta.shape[1]
        return np.dot(self.weights.transpose(), delta)

    def step(self, learning_rate):
        '''SGD Weight Update'''
        self.weights -= learning_rate * self.grad_w
        self.bias -= learning_rate * self.grad_b