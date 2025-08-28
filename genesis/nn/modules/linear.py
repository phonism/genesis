"""Linear (fully connected) layers."""

from typing import Optional
import genesis
from genesis import init
from ...autograd import Tensor
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
        self.weight = Parameter(
            init.randn(self.out_features, self.in_features, std=0.02),
            device=device, dtype=dtype)

        self.bias = None
        if bias:
            self.bias = Parameter(init.zeros(self.out_features), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the linear layer.
        """
        x = x @ self.weight.transpose(0, 1)
        if self.bias:
            x = x + self.bias
        return x


class Flatten(Module):
    """
    Flatten layer.
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the flatten layer.
        """
        return x.reshape(x.shape[0], -1)