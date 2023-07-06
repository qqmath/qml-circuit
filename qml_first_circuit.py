import sys
import pennylane as qml
import pandas as pd
import torch

from pennylane import numpy as np
from tqdm import tqdm
from collections import Counter
from pennylane.optimize import AdamOptimizer


NUM_WIRES = 4
NUM_LAYERS = 2                          #Number of embedding layers

device = qml.device("default.qubit", wires=NUM_WIRES)

def state_preparation(x):  

    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RX(x[2], wires=2)
    qml.RX(x[3], wires=3)
    pass


@qml.qnode(device, diff_method='adjoint')   #We declare a quantum node
def circuit(weights, x):

    state_preparation(x)                  #apply the state preparation protocole

    for i in range(NUM_LAYERS):           #apply random rotations
        for j in range(NUM_WIRES):          
            qml.Rot(*weights[i,j], wires=i)   

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])

    #We measure the 3 first wires by using a pauliZ measurement 
    return [qml.expval(qml.PauliZ(k)) for k in range(3)] 


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy(X,y):
   
    m = y.shape[0]
    p = np.array([softmax(x) for x in X])
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss


def accuracy(X,y):

    n = len(X)
    compteur = 0
    for k in range(len(X)):
        #Little argmax function
        index, max_val = -1, -1
        for i in range(len(X[k])):
            if X[k][i] > max_val:
                index, max_val = i, X[k][i]

        if index == y[k]:
            compteur += 1

    return compteur/n


def cost(weights,X, y):
    predictions = [circuit(weights, x) for x in X]  
    return cross_entropy(predictions, y)


def load_data(path: str ="~/data/iris.csv"):

    #importing the data with pandas read_csv function
    iris_data = pd.read_csv(path).drop('species', axis=1)
    iris_data = iris_data.to_numpy()      #convert to numpy

    data=[]
    for k in range(len(iris_data)):       #the data are classed by names 
        if k < 50:                          #by batches of 50
            data.append([iris_data[k],0])
        elif k < 100 and k >= 50:
            data.append([iris_data[k],1])
        elif k >= 100 and k < 150:
            data.append([iris_data[k],2])

    np.random.shuffle(data)
    X=[]
    y=[]
    for a,b in data:
        X.append(a)
        y.append(b)

    X = np.array(X)
    y = np.array(y)

    return X[:125], y[:125], X[125:], y[125:]    #spliting data into train/test


X_train, y_train, X_test, y_test = load_data()

#weights initialization for our quantum layers 
weights = (0.01 * np.random.randn(NUM_LAYERS, NUM_WIRES, 3))

#I've mad an easy batch loader it's not perfect, adapt with your train/test data
BATCH_SIZE = 5   
EPOCH = 10
LR = 1e-3

opt = AdamOptimizer(LR)


for epoch in tqdm(range(EPOCH), desc="Epoch"):
    with tqdm(range(0,len(X_train)-BATCH_SIZE,BATCH_SIZE),desc="Train") as pbar:

        for batch in pbar:
            X_train_batch = X_train[batch:batch+BATCH_SIZE]
            y_train_batch = y_train[batch:batch+BATCH_SIZE]

            weights = opt.step(
                lambda v: cost(v, X_train_batch, y_train_batch), weights
                )

            #Compute loss and accuracy
            predictions = [circuit(weights, x) for x in X_train]
            acc = accuracy(predictions, y_train)
            loss = cross_entropy(predictions, y_train)

            pbar.set_postfix(loss=f"{loss:.4f}", accuracy=f"{acc*100:.2f}%") 

    with tqdm(range(0,len(X_test)-BATCH_SIZE,BATCH_SIZE), desc="Test") as pbar:

        for batch in pbar:
            X_test_batch = X_test[batch:batch+BATCH_SIZE]
            y_test_batch = y_test[batch:batch+BATCH_SIZE]

            weights = opt.step(
                lambda v: cost(v, X_test_batch, y_test_batch), weights
                )

            #Compute loss and accuracy
            predictions = [circuit(weights, x) for x in X_test]
            acc = accuracy(predictions, y_test)
            loss = cross_entropy(predictions, y_test)

            pbar.set_postfix(loss=f"{loss:.4f}", accuracy=f"{acc*100:.2f}%")
