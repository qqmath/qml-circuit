# Quantum Neural Network for Iris Flower Classification

This code implements a Quantum Neural Network (QNN) for the classification of Iris flowers using PennyLane, a Python library for quantum machine learning. The QNN is designed to classify the Iris dataset into three categories based on the species of the flowers.
This QML-circuit was built using the tutorial  https://dvic.devinci.fr/en/resource/tutorial/introduction-to-quantum-machine-learning

## Prerequisites

Make sure you have the following libraries installed:

- PennyLane
- pandas
- numpy
- tqdm

You can install them using the following commands:

```bash
pip install pennylane pandas numpy tqdm
```

## Dataset

The Iris dataset is used for training and testing the QNN. It contains 150 samples of Iris flowers, each with four features (sepal length, sepal width, petal length, petal width), and belongs to one of three species (setosa, versicolor, virginica).

## Quantum Circuit Architecture

The Quantum Neural Network consists of two layers of random rotations and entangling CNOT gates. The input data is prepared using RX gates, and the predictions are made by measuring the first three qubits using PauliZ operators.

## Running the Code

1. First, import the required libraries:

```python
import sys
import pennylane as qml
import pandas as pd
import torch
from pennylane import numpy as np
from tqdm import tqdm
from collections import Counter
from pennylane.optimize import AdamOptimizer
```

2. Set the number of qubits and embedding layers:

```python
NUM_WIRES = 4
NUM_LAYERS = 2
```

3. Initialize the quantum device with four qubits:

```python
device = qml.device("default.qubit", wires=NUM_WIRES)
```

4. Define the quantum state preparation function and the quantum circuit using PennyLane's qnode decorator:

```python
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

    # Apply random rotations and entangling CNOT gates for each layer
    # ...
```

5. Implement the softmax, cross-entropy, and accuracy functions:

```python
# Softmax function to calculate the probability distribution
def softmax(x):
    # ...

# Cross-entropy loss function
def cross_entropy(X, y):
    # ...

# Accuracy function
def accuracy(X, y):
    # ...
```

6. Define the cost function for optimization:

```python
# Cost function for optimization
def cost(weights, X, y):
    predictions = [circuit(weights, x) for x in X]
    return cross_entropy(predictions, y)
```

7. Load the Iris dataset and split it into training and testing sets:

```python
def load_data(path: str = "~/data/iris.csv"):
    # ...

X_train, y_train, X_test, y_test = load_data()
```

8. Initialize the weights for the quantum layers and set hyperparameters for training:

```python
weights = (0.01 * np.random.randn(NUM_LAYERS, NUM_WIRES, 3))
BATCH_SIZE = 5
EPOCH = 10
LR = 1e-3
opt = AdamOptimizer(LR)
```

9. Run the training loop and test the model on the Iris dataset:

```python
for epoch in tqdm(range(EPOCH), desc="Epoch"):
    # Training loop
    # ...

    for batch in pbar:
        # ...

    # ...

    for batch in pbar:
        # ...
```

## Conclusion

This Quantum Neural Network provides a simple example of applying quantum computing to a classification task using the PennyLane library. The code can be further adapted and extended for more complex quantum models and datasets. Feel free to experiment with different hyperparameters and quantum circuit architectures to improve the classification performance. Happy quantum machine learning!
