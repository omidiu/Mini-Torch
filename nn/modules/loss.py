from .module import Module
from tensor import Tensor
from .. import functional as F


class _Loss(Module):
    reduction: str

    def __init__(self, reduction: str = 'mean') -> None:
        valid_reductions = {'mean', 'sum', 'none'}
        if reduction not in valid_reductions:
            raise ValueError(f"Invalid reduction: {reduction}. Expected one of {valid_reductions}.")
        self.reduction = reduction
        super(_Loss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(_Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)
