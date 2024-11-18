from .module import Module
from tensor import Tensor
import numpy as np
import nn.functional as F

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.random.rand(out_features, in_features))
        if bias:
            self.bias = Tensor(np.random.rand(1, out_features))
        else:
            self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


