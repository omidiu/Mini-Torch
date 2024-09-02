




import random
from .module import Module
from Tensor import Tensor





class Neuron(Module):

    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = [Tensor(random.random()) for _ in range(input_size)]
        self.bias = Tensor(random.random())

    def forward(self, input_data):
        return sum(w * x_i for w, x_i in zip(self.weights, input_data)) + self.bias

    def parameters(self):
        return self.weights + [self.bias]


class Layer(Module):

    def __init__(self, input_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP(Module):

    def __init__(self, *layer_sizes, **kwargs):
        super().__init__(*layer_sizes, **kwargs)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

