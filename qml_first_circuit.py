import sys
import pennylane as qml
import pandas as pd
import torch
from pennylane import numpy as np
from tqdm import tqdm
from collections import Counter
from pennylane.optimize import AdamOptimizer

NUM_WIRES = 4
NUM_LAYERS = 2  # Number of embedding layers

# Set up the quantum device with four qubits
device = qml.device("default.qubit", wires=NUM_WIRES)

# Quantum state preparation function
def state_preparation(x):
    # Apply RX gates on each qubit with input parameters x[0], x[1], x[2], x[3]
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RX(x[2], wires=2)
    qml.RX(x[3], wires=3)

# Quantum circuit definition using PennyLane's qnode decorator
@qml.qnode(device, diff_method='adjoint')
def circuit(weights, x):
    # Apply state preparation protocol to the quantum circuit
    state_preparation(x)

    for i in range(NUM_LAYERS):
        for j in range(NUM_WIRES):
            # Apply Rotations (RX, RY, RZ) with parameters from weights for each qubit
            qml.Rot(*weights[i, j], wires=i)

        # Apply CNOT gates between neighboring qubits
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])

    # Measure the first 3 qubits using a PauliZ measurement
    return [qml.expval(qml.PauliZ(k)) for k in range(3)]

# Softmax function to calculate the probability distribution
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# Cross-entropy loss function
def cross_entropy(X, y):
    m = y.shape[0]
    p = np.array([softmax(x) for x in X])
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss

# Accuracy function
def accuracy(X, y):
    n = len(X)
    compteur = 0
    for k in range(len(X)):
        # Find the index of the maximum value in X[k] (argmax function)
        index, max_val = -1, -1
        for i in range(len(X[k])):
            if X[k][i] > max_val:
                index, max_val = i, X[k][i]

        # Check if the predicted index matches the true label y[k]
        if index == y[k]:
            compteur += 1

    return compteur / n

# Cost function for optimization
def cost(weights, X, y):
    predictions = [circuit(weights, x) for x in X]
    return cross_entropy(predictions, y)

# Function to load data from the iris.csv file
def load_data(path: str = "~/data/iris.csv"):
    iris_data = pd.read_csv(path).drop('species', axis=1)
    iris_data = iris_data.to_numpy()

    data = []
    for k in range(len(iris_data)):
        if k < 50:
            data.append([iris_data[k], 0])
        elif k < 100 and k >= 50:
            data.append([iris_data[k], 1])
        elif k >= 100 and k < 150:
            data.append([iris_data[k], 2])

    np.random.shuffle(data)
    X = []
    y = []
    for a, b in data:
        X.append(a)
        y.append(b)

    X = np.array(X)
    y = np.array(y)

    return X[:125], y[:125], X[125:], y[125:]

# Load the data into training and testing sets
X_train, y_train, X_test, y_test = load_data()

# Initialize weights for the quantum layers
weights = (0.01 * np.random.randn(NUM_LAYERS, NUM_WIRES, 3))

# Set hyperparameters for training
BATCH_SIZE = 5
EPOCH = 10
LR = 1e-3

# Initialize the Adam optimizer
opt = AdamOptimizer(LR)

# Training loop
for epoch in tqdm(range(EPOCH), desc="Epoch"):
    with tqdm(range(0, len(X_train) - BATCH_SIZE, BATCH_SIZE), desc="Train") as pbar:
        for batch in pbar:
            X_train_batch = X_train[batch:batch + BATCH_SIZE]
            y_train_batch = y_train[batch:batch + BATCH_SIZE]

            # Perform one step of the Adam optimizer to update the weights
            weights = opt.step(lambda v: cost(v, X_train_batch, y_train_batch), weights)

            # Compute loss and accuracy for training set
            predictions = [circuit(weights, x) for x in X_train]
            acc = accuracy(predictions, y_train)
            loss = cross_entropy(predictions, y_train)

            # Update progress bar with the current loss and accuracy
            pbar.set_postfix(loss=f"{loss:.4f}", accuracy=f"{acc * 100:.2f}%")

    with tqdm(range(0, len(X_test) - BATCH_SIZE, BATCH_SIZE), desc="Test") as pbar:
        for batch in pbar:
            X_test_batch = X_test[batch:batch + BATCH_SIZE]
            y_test_batch = y_test[batch:batch + BATCH_SIZE]

            # Perform one step of the Adam optimizer to update the weights on the test set
            weights = opt.step(lambda v: cost(v, X_test_batch, y_test_batch), weights)

            # Compute loss and accuracy for test set
            predictions = [circuit(weights, x) for x in X_test]
            acc = accuracy(predictions, y_test)
            loss = cross_entropy(predictions, y_test)

            # Update progress bar with the current loss and accuracy
            pbar.set_postfix(loss=f"{loss:.4f}", accuracy=f"{acc * 100:.2f}%")
