"""Normalization layers."""

from typing import Optional
import genesis
from genesis import init
from genesis.tensor import Tensor
import genesis.nn.functional as F
from .module import Module, Parameter
from ..triton_ops import fused_layer_norm, fused_rmsnorm


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
            # Update running mean with exponential moving average
            momentum_term = self.momentum * mean_squeezed.detach()
            prev_term = (1 - self.momentum) * self.running_mean
            self.running_mean = (momentum_term + prev_term).detach()
            
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
        Forward pass of the layer normalization layer using fused Triton kernel.
        """
        if x.shape[-1] != self.dim:
            raise RuntimeError("Input dims should be %d" % self.dim)
        return fused_layer_norm(x, self.weight, self.bias, self.eps)


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
    RMS normalization layer with fused Triton kernel.

    Uses a single optimized kernel instead of decomposing into 7 separate operations,
    reducing kernel launch overhead and improving performance by ~7x for this operation.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Parameter(init.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the RMS normalization layer.

        Uses fused Triton kernel for CUDA, decomposed implementation for CPU.
        """
        # Use fused kernel only for CUDA tensors
        if x.device.is_cuda():
            return fused_rmsnorm(x, self.weight, self.eps)

        # Fallback decomposed implementation for CPU
        x_square = x ** 2
        x_mean = F.summation(x_square, axis=-1, keepdims=True) / x_square.shape[-1]
        rms = x / F.sqrt(x_mean + self.eps)
        return rms * self.weight
