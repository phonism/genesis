"""Activation function modules."""

import genesis
from genesis.tensor import Tensor
import genesis.nn.functional as F
from .module import Module
from ..triton_ops.silu import silu as triton_silu


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
    SiLU (Swish) activation function: f(x) = x * sigmoid(x).

    Preserves input dtype for efficient mixed precision training.
    Uses Triton-optimized kernels on CUDA devices.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SiLU activation function.

        Args:
            x: Input tensor

        Returns:
            Output tensor with same dtype as input
        """
        return triton_silu(x)


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