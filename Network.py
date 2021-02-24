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


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2

class MSE:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return (y_true - y_pred)**2

    def derivative(self, y_true, y_pred):
        return 2*np.mean(y_true - y_pred)

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
        self.grad_b = np.mean(delta, axis=1, keepdims=True)
        return np.dot(self.weights.transpose(), delta)

    def step(self, learning_rate):
        '''SGD Weight Update'''
        self.weights = self.weights - learning_rate * self.grad_w
        self.bias = self.bias - learning_rate * self.grad_b



class Network:
    def __init__(self, loss):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.loss = loss

    # Adding layer object to list self.layers
    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, mini_batch_size, learning_rate, shuffle=True):
        print('Start Training!')

        # Shuffle all training Data
        if shuffle == True:
            X = X
            y = y

        # Split whole Batch in Mini-Batches
        mini_batches_X = []
        mini_batches_y = []
        for i in range(X.shape[1]//mini_batch_size):
            X_mini_batch = X[:, i*mini_batch_size:(i+1)*mini_batch_size]
            y_mini_batch = y[:, i*mini_batch_size:(i+1)*mini_batch_size]
            mini_batches_X.append(X_mini_batch)
            mini_batches_y.append(y_mini_batch)

        # Training Loop
        for epoch in range(epochs):
            loss = []
            for XX, y_true in zip(mini_batches_X, mini_batches_y):

                # Forward Pass
                y_pred = self.predict(XX)

                # Backward Pass
                output_error = self.loss(y_true, y_pred)
                loss.append(np.mean(output_error))
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error)

                for layer in self.layers:
                    layer.step(learning_rate)

            print('Epoch ' + str(epoch+1) + '/' + str(epochs), 'Loss', np.mean(loss))


model = Network(loss=MSE())
model.add(Dense(100, 50, activation=Sigmoid()))
model.add(Dense(50, 10, activation=Sigmoid()))
model.add(Dense(10, 10, activation=Sigmoid()))
model.add(Dense(10, 5, activation=Sigmoid()))


X_train = np.random.random(size=(100, 20000))
y_train = np.random.random(size=(5, 20000))


model.fit(X_train, y_train, epochs=10, mini_batch_size=5000, learning_rate=0.001)