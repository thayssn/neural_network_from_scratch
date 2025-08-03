import numpy as np

def one_hot_encoding(x, possibilities):
    one_hot = np.zeros((x.size, possibilities))
    row = np.arange(x.size) 
    col = x
    one_hot[row, col] = 1
    return one_hot.T

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    x_max = np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def sig(x):
 return 1/(1 + np.exp(-x))

def deriv_relu(x):
    return x > 0

def deriv_sig(x):
    return sig(x) * (1 - sig(x))