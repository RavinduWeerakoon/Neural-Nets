import csv
import numpy as np

class BaseNeuralNetwork:
    def __init__(self, initial_weights=None, initial_biases=None, layer_sizes=None):
        self.layer_sizes = layer_sizes
        if layer_sizes:
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
                layer_sizes = []
                current_layer = None
                current_weights = []

                for row in reader:
                    layer_info = row[0]
                    parts = layer_info.split(' ')
                    layer_from = int(parts[2][5:])
                    layer_to = int(parts[4][5:])

                    if current_layer is None:
                        current_layer = layer_from
                        # layer_sizes.append(len(row) - 1)  # Input layer size

                    if current_layer != layer_from:
                        layer_sizes.append(len(current_weights))  # Previous layer size
                        weights.append(np.array(current_weights))
                        current_weights = []
                        current_layer = layer_from

                    current_weights.append(np.array(row[1:], dtype=np.float32))

                # Append the metadata about the last layer
                layer_sizes.append(len(current_weights))
                layer_sizes.append(len(current_weights[0]))
                #adding the num layers
                self.num_layers = len(layer_sizes)
                weights.append(np.array(current_weights))

                self.weights = weights
                self.layer_sizes = layer_sizes
#                print(self.layer_sizes)
            
            with open(biases_file, 'r') as bf:
                reader = csv.reader(bf)
                biases = []
                for row in reader:
                    layer_biases = np.array(row[1:], dtype=np.float32)
                    biases.append(layer_biases)
                    self.biases = biases

    def save_gradients(self, w,b):
        with open(w, 'w', newline='') as wf:
            writer = csv.writer(wf)
            for i in range(len(self.grad_w)):
                for x in self.grad_w[i]:
                    temp_row = x
                    writer.writerow(temp_row)
        with open(b, 'w', newline='') as bf:
            writer = csv.writer(bf)
            for i in range(len(self.grad_b)):
                temp_row = self.grad_b[i]
                writer.writerow(temp_row)
        
        
        
    def get_weights(self):
        print([s.shape for s in self.weights])
        return self.weights

    def get_biases(self):
        print([s.shape for s in self.biases])
        return self.biases