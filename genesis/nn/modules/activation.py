"""Activation function modules."""

import genesis
from genesis.tensor import Tensor
import genesis.nn.functional as F
from .module import Module


class ReLU(Module):
    """
    ReLU activation function.
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ReLU activation function.
        """
        x = genesis.relu(x)
        return x


class Softmax(Module):
    """
    Softmax activation function.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the softmax activation function.
        """
        if x.device == genesis.device('cpu') or genesis.use_triton is False:
            x_exp = F.exp(x - F.max(x, self.dim, keepdims=True))
            x = x_exp / F.summation(x_exp, axis=self.dim, keepdims=True)
            return x
        else:
            return F.softmax(x, dim=self.dim)


class SiLU(Module):
    """
    SiLU activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SiLU activation function.
        """
        return x / (F.exp(-x) + 1)


class Residual(Module):
    """
    Residual connection.
    """
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the residual connection.
        """
        return self.fn(x) + x