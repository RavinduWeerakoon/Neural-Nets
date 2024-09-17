import numpy as np
import csv
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

# Loss functions and their derivatives
def categorical_crossentropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

def categorical_crossentropy_derivative(y_true, y_pred):
    return y_pred - y_true


class BaseNeuralNetwork:
    def __init__(self, layer_sizes, initial_weights=None, initial_biases=None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.layer_initializer_from_initial(layer_sizes, initial_weights, initial_biases)
        self.gead_w = []
        self.grad_b = []
    
    def layer_initializer_from_initial(self, layer_sizes, weights_file, biases_file):
        # Initialize weights
        if weights_file is None:
            self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers - 1)]
            self.biases = [np.random.randn(i) for i in self.layer_sizes[1:]]
        else:
            with open(weights_file, 'r') as wf:
                reader = csv.reader(wf)
                weights = []
                for layer in layer_sizes[:-1]:
                    l_weights = []
                    for i in range(layer):
                        row = next(reader)
                        l_weights.append(np.array(row[1:], dtype=np.float32))
                    l_weights = np.array(l_weights)
                    weights.append(l_weights)
                self.weights = weights
            
            with open(biases_file, 'r') as bf:
                reader = csv.reader(bf)
                biases = []
                for row in reader:
                    layer_biases = np.array(row[1:], dtype=np.float32)
                    biases.append(layer_biases)
                    self.biases = biases

    def save_gradients(self, w,b):
        with open(w, 'w') as wf:
            writer = csv.writer(wf)
            for i in range(len(self.grad_w)):
                for x in self.grad_w[i]:
                    temp_row = x.tolist()
                    writer.writerow(temp_row)
        with open(b, 'w') as bf:
            writer = csv.writer(bf)
            for i in range(len(self.grad_b)):
                temp_row = self.grad_b[i].tolist()
                writer.writerow(temp_row)
        
        
        
    def get_weights(self):
        print([s.shape for s in self.weights])
        return self.weights

    def get_biases(self):
        print([s.shape for s in self.biases])
        return self.biases

class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, layer_sizes, initial_weights=None, initial_biases=None):
        super().__init__(layer_sizes, initial_weights, initial_biases)

        

    
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
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            activations = self.forward_propagation(X)
            self.backward_propagation(X, y, activations, learning_rate)
            if epoch % 100 == 0:
                loss = categorical_crossentropy(y, activations[-1])
                print(f'Epoch {epoch}, Loss: {loss}')

# def read_data_point(filepath):


# Example usage
if __name__ == "__main__":
    # Example dataset (X: inputs, y: one-hot encoded outputs)
    # X = np.array([[0, 0, 1,1,1,1,1,1,0,0,0,1,0,1],
    #               [0, 1, 1,1,1,1,1,1,0,0,0,1,0,1]])
    # y = np.array([[1, 0, 0, 0],
    #               [0, 1, 0, 0]])
    X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]], dtype=np.float32)
    y = np.array([[0, 0, 0, 1]], dtype=np.float32)

    # Define the network architecture
    layer_sizes = [14, 100, 40, 4]  # Input layer, two hidden layers, output layer with 4 neurons

    # Training parameters
    epochs = 10
    learning_rate = 0.1

    # Create and train the neural network
    nn = NeuralNetwork(layer_sizes, "Assignment_1/Task_1/a/w.csv" , "Assignment_1/Task_1/a/b.csv")
    activations = nn.forward_propagation(X)
    nn.backward_propagation(X, y, activations, learning_rate)
    # print(len(nn.grad_w))
    print(nn.grad_b)
