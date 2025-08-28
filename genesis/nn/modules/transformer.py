"""Transformer architecture components."""

from typing import Optional
import numpy as np
import genesis
from genesis import init
from ...autograd import Tensor
import genesis.nn.functional as F
from .module import Module, Parameter
from .linear import Linear
from .activation import SiLU, Softmax
from .dropout import Dropout


class FeedFowardSwiGLU(Module):
    """ 
    SwiGLU: https://arxiv.org/pdf/2002.05202.pdf
    """
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int
    ):
        super().__init__()
        self.gate = Linear(dim, hidden_dim, bias=False)
        self.down = Linear(hidden_dim, dim, bias=False)
        self.up = Linear(dim, hidden_dim, bias=False)
        self.act = SiLU()
        self.dropout = Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the feed forward SwiGLU layer.
        """
        out = self.down(self.act(self.gate(x)) * self.up(x))
        return self.dropout(out)


class MultiheadAttention(Module):
    """
    Multihead attention layer.
    """
    def __init__(
        self, 
        dim: int = 64, 
        heads: int = 1, 
        device: Optional[str] = None, 
        dtype: Optional[str] = "float32"
    ):
        self.dim = dim
        self.heads = heads
        self.w_qkv = Parameter(
            init.kaiming_uniform(self.dim, self.dim * 3),
            device=device, dtype=dtype)
        self.w_out = Parameter(
            init.kaiming_uniform(self.dim, self.dim),
            device=device, dtype=dtype)
        self.softmax = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the multihead attention layer.
        """
        q, k, v = F.split((x @ self.w_qkv).reshape(x.shape[0], x.shape[1], 3, self.dim), dim=2)
        q, k, v = [a.reshape(x.shape[0], x.shape[1], self.heads, self.dim // self.heads).transpose((1, 2)) for a in [q, k, v]]
        mask = genesis.triu((-float("inf") * init.ones(x.shape[1], x.shape[1], device=x.device)), k=1, device=x.device)
        atten = self.softmax(q @ F.transpose(k) / np.sqrt(self.dim // self.heads) + mask)
        return (atten @ v).transpose((1, 2)).reshape(x.shape[0], x.shape[1], self.dim) @ self.w_out, atten


class FusedMultiheadAttention(Module):
    """
    Fused multihead attention layer.
    """
    def __init__(
        self, 
        dim: int = 64, 
        heads: int = 1, 
        device: Optional[str] = None, 
        dtype: Optional[str] = "float32"
    ):
        self.dim = dim
        self.heads = heads
        self.w_qkv = Parameter(
            init.kaiming_uniform(self.dim, self.dim * 3),
            device=device, dtype=dtype)
        self.w_out = Parameter(
            init.kaiming_uniform(self.dim, self.dim),
            device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the fused multihead attention layer.
        """
        q, k, v = F.split((x @ self.w_qkv).reshape(x.shape[0], x.shape[1], 3, self.dim), dim=2)
        q, k, v = [a.reshape(x.shape[0], x.shape[1], self.heads, self.dim // self.heads).transpose((1, 2)) for a in [q, k, v]]
        return F.fused_attention(q, k, v).transpose((1, 2)).reshape(x.shape[0], x.shape[1], self.dim) @ self.w_out, None