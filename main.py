import pandas as pd
import numpy as np
import utils as u
import matplotlib.pyplot as plt


def init_params(input):
    rows, _ = input.shape
    w1 = np.random.rand(10, rows) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def forward_propagation(input, w1, b1, w2, b2):
    z1 = w1.dot(input) + b1
    a1 = u.relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = u.softmax(z2)
    return z1, a1, z2, a2


def backward_propagation(input, z1, a1, z2, a2, w1, b1, w2, b2, one_hot_outputs):
    _,samples_size = input.shape
    dz2 = a2 - one_hot_outputs
    dw2 = 1 / samples_size * dz2.dot(a1.T)
    db2 = 1 / samples_size * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * u.deriv_relu(z1)
    dw1 = 1 / samples_size * dz1.dot(input.T)
    db1 = 1 / samples_size * np.sum(dz1)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    return w1, b1, w2, b2 

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / len(labels)

def gradient_descent(input, iterations, learning_rate, output):
    w1, b1, w2, b2 = init_params(input)
    one_hot_labels = u.one_hot_encoding(output, 10)
    accuracy_list = []
    for i in range(iterations):
        z1,a1,z2,a2 = forward_propagation(input, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward_propagation(input,z1,a1,z2,a2,w1,b1,w2,b2, one_hot_labels)
        w1,b1,w2,b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)
        predictions = get_predictions(a2)
        accuracy = get_accuracy(predictions, output)
        accuracy_list.append(accuracy)
        if i % 10 == 0:
            plt.title(f'accuracy: {accuracy} iteration: {i}')
            plt.clf()
            plt.plot(accuracy_list)
    print('end of training')
    return w1, b1, w2, b2
