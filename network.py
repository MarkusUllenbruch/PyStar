import numpy as np

class Sequential:
    def __init__(self, loss):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.loss = loss

    # Adding layer object to list self.layers
    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X, train=True):

        if train == False:
            output = X.transpose()
        else:
            output = X

        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, mini_batch_size, learning_rate, shuffle=True):
        print('Start Training!')
        X = X.transpose()
        y = y.transpose()

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
                #print(y_pred)

                # Backward Pass
                output_error = self.loss.derivative(y_true, y_pred)
                loss.append(np.mean(self.loss.derivative(y_true, y_pred)))
                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error)

                for layer in self.layers:
                    layer.step(learning_rate)

            print('Epoch ' + str(epoch+1) + '/' + str(epochs), 'Loss', np.mean(loss))