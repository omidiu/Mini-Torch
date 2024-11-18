class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = 0

    def step(self, closure=None):
        raise NotImplementedError
