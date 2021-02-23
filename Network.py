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

class MSE:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

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
            for XX, y_true in zip(mini_batches_X, mini_batches_y):

                # Forward Pass
                y_pred = self.predict(XX)

                # Backward Pass
                errors = []
                delta = self.loss(y_true, y_pred) * self.layers[-1].z
                errors.append(delta)

                for i in range(2, len(self.layers)+1):
                    o = self.layers[-i].a
                    print(o)
                #    delta = np.dot(self.layers[i+1].weights.transpose(), delta) * self.layers[i].activation(self.layers[i].z)
                #    errors.append(delta)


            print('Epoch ' + str(epoch+1) + '/' + str(epochs))

    def backprop(self, x, y):
        pass


model = Network(loss=MSE())
model.add(Dense(100, 50, activation=Linear()))
model.add(Dense(50, 10, activation=Linear()))
model.add(Dense(10, 10, activation=Linear()))
model.add(Dense(10, 5, activation=Linear()))


X_train = np.random.random(size=(100, 1000))
y_train = np.random.random(size=(5, 1000))


model.fit(X_train, y_train, epochs=1, mini_batch_size=1000, learning_rate=0.001)