class Layer:

    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, input_data):
        # pass the input through each neuron and collect outputs
        return [neuron.forward(input_data) for neuron in self.neurons]
        
    def parameters(self): 
        # gather parameters from all neurons in this layer
        return [param for neuron in self.neurons for param in neuron.parameters()]
