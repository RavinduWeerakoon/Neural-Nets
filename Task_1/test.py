import csv
import numpy as np

class BaseNeuralNetwork:
    def __init__(self, layer_sizes, initial_weights=None, initial_biases=None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.layer_initializer_from_initial(layer_sizes, initial_weights, initial_biases)
        self.grad_w = []
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
                        l_weights.append(np.array(row[1:], dtype=float))
                    l_weights = np.array(l_weights)
                    weights.append(l_weights)
                self.weights = weights
            
            with open(biases_file, 'r') as bf:
                reader = csv.reader(bf)
                biases = []
                for row in reader:
                    layer_biases = np.array(row[1:], dtype=float)
                    biases.append(layer_biases)
                    self.biases = biases

    def save_network(self, weights_file, biases_file):
        with open(weights_file, 'w') as wf:
            writer = csv.writer(wf)
            for i in range(len(self.weights)):
                for x in self.weights[i]:
                    temp_row = x.tolist()
                    temp_row.insert(0, f"weights btw layer{i} to layer{i+1}")
                    writer.writerow(temp_row)
            
        with open(biases_file, 'w') as bf:
            writer = csv.writer(bf)
            for i in range(len(self.biases)):
                temp_row = self.biases[i].tolist()
                temp_row.insert(0, f"biases for layer{i+1}")
                writer.writerow(temp_row)

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

if __name__ == "__main__":
    layer_sizes = [14, 100, 40, 4]
    #nn = BaseNeuralNetwork(layer_sizes, 'Assignment_1/Task_1/a/w.csv', 'Assignment_1/Task_1/a/b.csv') 
    #nn.layer_initializer_from_initial(layer_sizes, 'Assignment_1/Task_1/a/w.csv', 'Assignment_1/Task_1/a/b.csv')
    nn = BaseNeuralNetwork(layer_sizes)
    nn.save_network("w.csv", "b.csv")
    
    # print(nn.get_weights())
    nn.get_weights()
    nn.get_biases()