"""Normalization layers."""

from typing import Optional
import genesis
from genesis import init
from ...autograd import Tensor
import genesis.nn.functional as F
from .module import Module, Parameter


class BatchNorm1d(Module):
    """
    Batch normalization layer.
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: Optional[str] = None,
        dtype: Optional[str] = "float32"
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the batch normalization layer.
        """
        if self.training:
            batch = x.shape[0]
            # Use keepdims to maintain shape consistency
            mean = F.summation(x, axis=0, keepdims=True) / batch
            mean_squeezed = mean.squeeze(0)
            self.running_mean = (self.momentum * mean_squeezed.detach() + (1 - self.momentum) * self.running_mean).detach()
            
            # Compute variance with keepdims
            centered = x - mean
            var = F.summation(centered ** 2, axis=0, keepdims=True) / batch
            var_squeezed = var.squeeze(0)
            self.running_var = (self.momentum * var_squeezed.detach() + (1 - self.momentum) * self.running_var).detach()
            
            # Normalize
            x_normalized = centered / F.sqrt(var + self.eps)
        else:
            mean = self.running_mean
            var = self.running_var
            x_normalized = (x - mean) / F.sqrt(var + self.eps)
        
        # Scale and shift
        x = self.weight * x_normalized + self.bias
        return x


class LayerNorm(Module):
    """
    Layer normalization layer.
    """
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Optional[str] = None,
        dtype: Optional[str] = "float32"
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the layer normalization layer.
        """
        if x.shape[-1] != self.dim:
            raise RuntimeError("Input dims should be %d" % self.dim)
        mean = F.summation(x, axis=-1, keepdims=True) / x.shape[-1]
        var = F.summation((x - mean) ** 2, axis=-1, keepdims=True) / self.dim
        output = (x - mean) / F.sqrt(var + self.eps)
        output = self.weight * output + self.bias
        return output


class FusedLayerNorm(Module):
    """
    Fused layer normalization layer.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the fused layer normalization layer.
        """
        return F.fused_layer_norm(x, self.weight, self.bias, self.eps)


class RMSNorm(Module):
    """
    RMS normalization layer.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Parameter(init.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the RMS normalization layer.
        """
        x_square = x ** 2
        x_mean = F.summation(x_square, axis=-1, keepdims=True) / x_square.shape[-1]
        rms = x / F.sqrt(x_mean + self.eps)
        return rms * self.weight