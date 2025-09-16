"""Dropout layers."""

from genesis import init
from genesis.tensor import Tensor
from .module import Module


class Dropout(Module):
    """
    Dropout layer.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the dropout layer.
        """
        if self.training and self.p > 0.0:
            mask = init.randb(*x.shape, p=(1 - self.p), dtype=x.dtype, device=x.device)
            x = x * mask / (1 - self.p)
        return x