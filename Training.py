import numpy as np
from activations import Sigmoid, Tanh
from layers import Dense
from losses import MSE
from network import Sequential

# Define Training Data
X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = np.array([[0.0], [0.0], [0.0], [1.0]])

# Initialize my own Network with MSE Lossfunction
model = Sequential(loss=MSE())

# Adding Layers to model
model.add(Dense(2, 5, activation=Sigmoid()))
model.add(Dense(5, 1, activation=Sigmoid()))

# Start training process
model.fit(X_train, y_train, epochs=1000, mini_batch_size=4, learning_rate=2)
pred = model.predict(X_train, train=False)
print(pred)