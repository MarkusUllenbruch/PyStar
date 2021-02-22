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
        self.weights = np.random.rand(input_neurons, output_neurons) - 0.5
        #print('WEIGHTS  ',self.weights.shape)
        self.bias = np.random.rand(1, output_neurons) - 0.5

    def forward(self, inputs):

        self.input = inputs
        self.z = np.dot(self.input, self.weights) + self.bias
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
        for i in range(X.shape[0]//mini_batch_size):
            print(i)
            X_mini_batch = X[i*mini_batch_size:(i+1)*mini_batch_size, :]
            y_mini_batch = y[i*mini_batch_size:(i+1)*mini_batch_size, :]
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

                print('LAYERS:', len(self.layers))

                for i in range(len(self.layers)-2, -1, -1):
                    print(i)
                    delta = np.dot(self.layers[i+1].weights.transpose(), delta) * self.layers[i].activation(self.layers[i].z)
                    errors.append(delta)





            print('Epoch ' + str(epoch+1) + '/' + str(epochs))

    def backprop(self, x, y):
        pass


model = Network(loss=MSE())
model.add(Dense(2, 10, activation=Linear()))
model.add(Dense(10, 10, activation=Linear()))
model.add(Dense(10, 10, activation=Linear()))
model.add(Dense(10, 1, activation=Linear()))


X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0, 0], [0, 0], [0, 0], [1, 0]])

Y = model.predict(X_train)
print(Y.shape, Y)

model.fit(X_train, y_train, epochs=10, mini_batch_size=2, learning_rate=0.001)