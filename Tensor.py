import numpy as np

class Tensor:

    def __init__(self, value, label='', children=(), operator=None):
        self.value = value
        self.children = set(children)
        self.operator = operator
        self.grad = 0
        self._backward = lambda: None
        self.label = label

    def __repr__(self) -> str:
        return f"Tensor(value={self.value})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.value + other.value, children=(self, other), operator='+')

        def backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.value * other.value, children=(self, other), operator='*')

        def backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        
        out._backward = backward

        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.value ** other.value, children=(self, other), operator='**')

        def backward():
            self.grad += (other.value * (self.value ** (other.value - 1))) * out.grad
            other.grad += (self.value ** other.value) * np.log(np.maximum(self.value, np.finfo(float).eps)) * out.grad

            # other.grad += (self.value ** other.value) * np.log(self.value) * out.grad

        out._backward = backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):
        return self + other

    def backward(self):
        topo_order = []
        visited = set()
        
        def topological_sort(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor.children:
                    topological_sort(child)
                topo_order.append(tensor)
        topological_sort(self)

        # initialize gradient of current node
        self.grad = 1.0
        
        for tensor in reversed(topo_order):
            tensor._backward()
