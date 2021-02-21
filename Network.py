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

def linear(x):
    return x

def linear_prime(x):
    return 1.0

class Dense(Layer):
    def __init__(self, input_neurons, output_neurons, activation):
        super().__init__()
        self.activation = activation
        self.weights = np.random.rand(input_neurons, output_neurons) - 0.5
        self.bias = np.random.rand(1, output_neurons) - 0.5

    def forward(self, inputs):
        # inputs muss size haben: (1, input_neurons)

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




model = Network()
model.add(Dense(100, 50, activation=linear))
model.add(Dense(50, 2, activation=linear))


batch = np.random.randint(low=-2, high=2, size=(5,100))
Y = model.predict(batch)
print(Y)