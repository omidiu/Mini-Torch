import numpy as np

from tensor import Tensor


def linear(input, weight, bias=None):
    # Ensure input and weight are Tensors
    input = input if isinstance(input, Tensor) else Tensor(input)
    weight = weight if isinstance(weight, Tensor) else Tensor(weight)
    bias = bias if isinstance(bias, Tensor) else Tensor(bias) if bias is not None else None

    input_flat = input.data.reshape(-1, input.data.shape[-1])  # (N*..., in_features)

    # y = xW^T (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
    output_data = np.dot(input_flat, weight.data.T)  # (N*..., out_features)

    output_shape = list(input.data.shape[:-1]) + [weight.data.shape[0]]
    output_data = output_data.reshape(output_shape)

    if bias is not None:
        output_data += bias.data

    out = Tensor(output_data, children=(input, weight, bias) if bias is not None else (input, weight),
                 operator='linear')

    def _backward():
        grad_flat = out.grad.reshape(-1, weight.data.shape[0])  # (N*..., out_features)
        input.grad += np.dot(grad_flat, weight.data).reshape(input.data.shape)  # Reshape back to input shape
        weight.grad += np.dot(grad_flat.T, input_flat)  # (out_features, in_features)
        if bias is not None:
            bias.grad += grad_flat.sum(axis=0)

    out._backward = _backward

    return out


def tanh(input):
    out = Tensor(np.tanh(input.data), children=(input,), operator='tanh')

    def _backward():
        input.grad += (1. - out.data ** 2) * out.grad

    out._backward = _backward
    return out


def relu(input):
    out = Tensor(np.maximum(0, input.data), children=(input,), operator="relu")

    def _backward():
        input.grad += (np.where(out.data > 0), 1, 0) * out.grad

    out._backward = _backward
    return out


def leaky_relu(input, alpha=0.001):
    out = Tensor(np.maximum(alpha * input.data, input.data), children=(input,), operator="relu")

    def _backward():
        input.grad += (np.where(out.data > 0), 1, alpha) * out.grad

    out._backward = _backward
    return out


def sigmoid(input):
    out = Tensor(1. / (1. + np.exp(- input.data)), children=(input,), operator="leaky_relu")

    def _backward():
        input.grad += (out.data * (1. - out.data)) * out.grad

    out._backward = _backward
    return out


def mse_loss(input, target, reduction):
    # TODO: Check Broadcasting
    squared_error = (input - target) ** 2
    if reduction == "mean":
        return squared_error.mean()  # Mean reduction
    elif reduction == "sum":
        return squared_error.sum()  # Sum reduction
    else:
        return squared_error


