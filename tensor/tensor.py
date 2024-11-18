import numpy as np

from utils.broadcasting import align_grad_shape


class Tensor:

    def __init__(self, value, label='', children=(), operator=None):
        self.data = np.array(value, dtype=np.float32) if not isinstance(value, np.ndarray) else value
        self.shape = self.data.shape
        self.children = set(children)
        self.operator = operator
        self.grad = np.zeros_like(self.data, dtype=self.data.dtype)
        self._backward = lambda: None
        self.label = label

    def __repr__(self) -> str:
        return f"Tensor(value={self.data})"

    def __pow__(self, other):  # TODO: CHECK THIS
        other = other if isinstance(other, Tensor) else Tensor(other)  # grad to false
        out = Tensor(self.data ** other.data, children=(self,), operator='**')

        def _backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, children=(self, other), operator='+')

        def _backward():
            self.grad += align_grad_shape(out.grad, self.shape)
            other.grad += align_grad_shape(out.grad, other.shape)

        out._backward = _backward

        return out

    def sum(self):
        out = Tensor(self.data.sum(), children=(self,), operator='sum')

        def backward():
            self.grad = out.grad * np.ones_like(self.data)

        out._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, children=(self, other), operator='*')

        def _backward():
            self.grad += (other.data * np.ones_like(self.data)) * out.grad

            if other.shape == ():
                other.grad += np.sum(self.data * out.grad)
            else:
                other.grad += np.sum((self.data * out.grad), axis=tuple(range(len(out.grad.shape) - len(other.shape))))

        out._backward = _backward

        return out

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
        self.grad = np.ones_like(self.data, dtype=self.data.dtype)

        for tensor in reversed(topo_order):
            tensor._backward()

    def __neg__(self):  # -self
        out = Tensor(-self.data, children=(self,), operator='neg')

        def backward():
            self.grad += -1 * out.grad

        out._backward = backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):  #
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1
