import numpy as np

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    total_predictions = y_true.shape[0]
    return correct_predictions / total_predictions


# Loss functions and their derivatives
def categorical_crossentropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

def categorical_crossentropy_derivative(y_true, y_pred):
    return y_pred - y_true

