import numpy as np
from utils import *
from base import BaseNeuralNetwork



class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, layer_sizes, initial_weights=None, initial_biases=None):
        super().__init__(layer_sizes, initial_weights, initial_biases)
        self.history = {}

        

    
    def forward_propagation(self, X):
        activations = [X]
        for i in range(self.num_layers - 1):
            net_input = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i == self.num_layers - 2:
                activation = softmax(net_input)  # Use softmax in the output layer
            else:
                activation = relu(net_input)
            activations.append(activation)
        return activations
    
    def backward_propagation(self, X, y, activations, learning_rate):
        deltas = [categorical_crossentropy_derivative(y, activations[-1])]
        
        for i in range(self.num_layers - 2, 0, -1):
            delta = deltas[-1].dot(self.weights[i].T) * relu_derivative(activations[i])
            deltas.append(delta)
        
        deltas.reverse()

        self.grad_w = []
        self.grad_b = []
        
        for i in range(self.num_layers - 1):
            self.grad_w.append(activations[i].T.dot(deltas[i]))
            self.grad_b.append(np.sum(deltas[i], axis=0))
            self.weights[i] -= activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] -= np.sum(deltas[i], axis=0) * learning_rate
    
    def train(self, X, y, epochs, learning_rate, testX=None, testY=None):
        training_loss = []
        testing_loss = []
        train_accuracy = []
        test_accuracy = []
        for epoch in range(epochs):
            activations = self.forward_propagation(X)
            self.backward_propagation(X, y, activations, learning_rate)
            loss = categorical_crossentropy(y, activations[-1])
            training_loss.append(loss)
           
            train_accuracy.append(accuracy_score(y, activations[-1]))
            if testX:
                act = self.forward_propagation(testX)[-1]
                testing_loss.append(categorical_crossentropy(testY,act ))
                test_accuracy.append(accuracy_score(testY,act))
            # self.history["training_loss"].append(loss)
            print(f'Epoch {epoch}, Loss: {loss}')

        self.history["training_loss"] = training_loss
        self.history["testing_loss"] = testing_loss
        self.history["train_accuracy"] = train_accuracy
        self.history["test_accuracy"] = test_accuracy

# def read_data_point(filepath):


# Example usage
if __name__ == "__main__":

    X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    y = np.array([[0, 0, 0, 1]])

    # Define the network architecture
    #layer_sizes = [14, 100, 40, 4]  #Input layer, two hidden layers, output layer with 4 neurons

    # Training parameters
    epochs = 10
    learning_rate = 0.1

    # Create and train the neural network
    nn = NeuralNetwork("NN/Task_1/b/w-100-40-4.csv" , "NN/Task_1/b/b-100-40-4.csv")
    activations = nn.forward_propagation(X)
    
    # #print(activations)
    nn.backward_propagation(X, y, activations, learning_rate)
    nn.save_gradients("dw.csv", "db.csv")
    # # print(len(nn.grad_w))
    # # print(nn.grad_b)
