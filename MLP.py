class MLP:

    def __init__(self, *layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, input_data):
        # pass the input through each layer in the MLP
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def parameters(self):
        # gather parameters from all layers
        return [param for layer in self.layers for param in layer.parameters()]
