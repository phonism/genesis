"""Sparse layers (Embedding)."""

from typing import Optional, Tuple
import numpy as np
import genesis
from genesis import init
from genesis.tensor import Tensor
import genesis.nn.functional as F
from .module import Module, Parameter


class Embedding(Module):
    """
    Embedding layer.
    """
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, std=0.02))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the embedding layer.
        """
        # Use direct indexing instead of one_hot + matmul
        # This is MUCH faster for large vocabularies
        return self.weight[x]


class RotaryEmbedding(Module):
    """
    Rotary embedding layer.
    """
    def __init__(
        self,
        dim,
        max_position_embeddings: int = 2048,
        base: int = 10000
    ):
        super().__init__()
        inv_freq_data = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.inv_freq = genesis.tensor(inv_freq_data)
        self.max_seq_len_cached = max_position_embeddings
        t_data = np.arange(self.max_seq_len_cached, dtype=np.float32)
        t = genesis.tensor(t_data)
        t = t.reshape(t.shape[0], 1)
        self.inv_freq = self.inv_freq.reshape(1, self.inv_freq.shape[0])
        freqs = t @ self.inv_freq
        emb = F.stack((freqs, freqs), dim=-1).transpose(-1, -2).reshape(freqs.shape[0], freqs.shape[1] * 2)
        self.cos_cached = emb.cos().reshape((1, 1) + (emb.shape))
        self.sin_cached = emb.sin().reshape((1, 1) + (emb.shape))

    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the rotary embedding layer.
        """
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )