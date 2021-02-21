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


class Linear:
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def derivative(self):
        return 1.0

class Dense(Layer):
    def __init__(self, input_neurons, output_neurons, activation):
        super().__init__()
        self.activation = activation
        self.weights = np.random.rand(input_neurons, output_neurons) - 0.5
        self.bias = np.random.rand(1, output_neurons) - 0.5

    def forward(self, inputs):

        self.input = inputs
        self.z = np.dot(self.input, self.weights) + self.bias
        self.a = self.activation(self.z)
        return self.a


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # Adding layer object to list self.layers
    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, mini_batch_size, learning_rate):
        pass

    def backprop(self, x, y):
        pass


model = Network()
model.add(Dense(100, 50, activation=Linear()))
model.add(Dense(50, 50, activation=Linear()))
model.add(Dense(50, 2, activation=Linear()))


batch = np.random.randint(low=-2, high=2, size=(5,100))
Y = model.predict(batch)
print(Y.shape)