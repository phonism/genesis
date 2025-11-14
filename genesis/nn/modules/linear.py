"""Linear (fully connected) layers."""

from typing import Optional
import genesis
from genesis import init
from genesis.tensor import Tensor
from .module import Module, Parameter


class Linear(Module):
    """
    Linear layer.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[str] = "float32"
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Pass device and dtype to tensor creation, not Parameter
        self.weight = Parameter(
            init.randn(self.out_features, self.in_features, std=0.02, device=device, dtype=dtype))

        self.bias = None
        if bias:
            self.bias = Parameter(init.zeros(self.out_features, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the linear layer.

        Implementation matches PyTorch's nn.Linear behavior under AMP:
        - Matmul uses FP16 in autocast mode (via Matmul.amp_policy = FP16)
        - Bias is converted to match matmul output dtype before addition
        - This prevents FP16 + FP32 → FP32 promotion, preserving FP16 acceleration
        """
        out = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            # Match PyTorch behavior: convert bias to matmul output dtype
            # This prevents mixed-dtype PROMOTE (FP16 + FP32 → FP32)
            # Bias is small (1D), conversion overhead is negligible
            if self.bias.dtype != out.dtype:
                bias = self.bias.to(out.dtype)
            else:
                bias = self.bias
            out = out + bias
        return out


class Flatten(Module):
    """
    Flatten layer.
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the flatten layer.
        """
        return x.reshape(x.shape[0], -1)