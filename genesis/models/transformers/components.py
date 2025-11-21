"""Shared components for Transformer models.

This module provides common building blocks used across different transformer
architectures (dense, MoE, etc.), including:
- Rotary position embeddings (RoPE)
- Position embedding utilities
- Common helper functions

Backend Selection:
    Supports both Genesis and PyTorch backends via NANOCHAT_BACKEND env var.
"""

from typing import Optional, Tuple
import numpy as np
import os

# Backend selection: check NANOCHAT_BACKEND environment variable
BACKEND = os.environ.get("NANOCHAT_BACKEND", "genesis")

if BACKEND == "torch":
    import torch as genesis
    from torch import Tensor
    import torch.nn as nn
else:
    import genesis
    from genesis import Tensor
    import genesis.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    Implements the rotary position embedding mechanism that encodes
    absolute position information with rotation matrices, allowing
    the model to generalize to longer sequences.

    Args:
        dim: Dimension of the rotary embeddings (usually head_dim).
        max_position_embeddings: Maximum sequence length to precompute.
        base: Base period for the rotation frequencies.
        device: Device to place the embeddings on.

    Attributes:
        inv_freq: Inverse frequencies for computing rotary embeddings.
        cos_cached: Precomputed cosine values.
        sin_cached: Precomputed sine values.
        max_seq_len_cached: Maximum sequence length cached.

    Reference:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.inv_freq = genesis.tensor(inv_freq, device=device)

        # Precompute embeddings for max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[str] = None):
        """
        Precompute and cache cosine and sine values for rotary embeddings.

        Args:
            seq_len: Sequence length to precompute embeddings for.
            device: Device to place the cached values on.
        """
        self.max_seq_len_cached = seq_len
        t = genesis.tensor(np.arange(seq_len, dtype=np.float32), device=device)
        t = t.reshape(t.shape[0], 1)
        inv_freq = self.inv_freq.reshape(1, self.inv_freq.shape[0])

        # Compute outer product: t @ inv_freq
        freqs = t @ inv_freq

        # Expand frequencies: [seq_len, dim/2] -> [seq_len, dim]
        emb = genesis.stack((freqs, freqs), dim=-1)
        emb = emb.transpose(-1, -2).reshape(freqs.shape[0], freqs.shape[1] * 2)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Get rotary embeddings for the given sequence length.

        Args:
            x: Input tensor (used for device/dtype reference).
            seq_len: Sequence length to get embeddings for. If None, uses x.shape.

        Returns:
            Tuple of (cos, sin) tensors for rotary embeddings.
        """
        if seq_len is None:
            seq_len = x.shape[-2] if len(x.shape) > 1 else x.shape[0]

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the hidden dimensions for RoPE.

    This function is used in the application of rotary position embeddings.
    It rotates the vector by swapping and negating half of its dimensions.

    Args:
        x: Input tensor with shape (..., hidden_dim).

    Returns:
        Tensor with the last half of dimensions negated and swapped with the first half.

    Example:
        >>> x = tensor([a, b, c, d])
        >>> rotate_half(x)  # returns [-c, -d, a, b]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return genesis.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    This function applies RoPE to both query and key tensors, which allows
    the model to encode relative position information through rotation.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim).
        k: Key tensor of shape (batch, heads, seq_len, head_dim).
        cos: Cosine values for rotary embedding.
        sin: Sine values for rotary embedding.
        position_ids: Position indices for each token. If None, uses sequential positions.
        unsqueeze_dim: Dimension to unsqueeze cos/sin tensors.

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied.

    Note:
        The rotation is applied as: q * cos + rotate_half(q) * sin
        This preserves the norm of the vectors while encoding position.
    """
    if position_ids is not None:
        cos = cos[position_ids.long()].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids.long()].unsqueeze(unsqueeze_dim)
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def find_multiple(n: int, k: int) -> int:
    """Find the smallest multiple of k that is >= n.

    This is useful for aligning sequence lengths to specific boundaries
    for efficiency (e.g., making sequence length a multiple of 8).

    Args:
        n: Target number.
        k: Multiple to align to.

    Returns:
        Smallest multiple of k that is >= n.

    Example:
        >>> find_multiple(10, 8)
        16
        >>> find_multiple(16, 8)
        16
        >>> find_multiple(7, 4)
        8
    """
    if n % k == 0:
        return n
    return n + k - (n % k)
