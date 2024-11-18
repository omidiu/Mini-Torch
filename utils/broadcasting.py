def align_grad_shape(grad, original_shape):
    extra_dims = grad.ndim - len(original_shape)
    for _ in range(extra_dims):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(original_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
