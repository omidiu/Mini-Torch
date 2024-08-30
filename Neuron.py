import random

class Neuron:

    def __init__(self, input_size):
        self.weights = [random.random() for _ in range(input_size)]
        self.bias = random.random()

    def forward(self, input_data):
        # weighted sum of inputs plus bias
        return sum(w * x_i for w, x_i in zip(self.weights, input_data)) + self.bias
        
    def parameters(self): 
        return self.weights + [self.bias]
