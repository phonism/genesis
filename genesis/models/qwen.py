"""Qwen model implementation in Genesis framework.

This module implements the Qwen (Tongyi Qianwen) language model architecture,
a transformer-based model with rotary position embeddings, multi-head attention,
and feed-forward networks. Supports various model sizes from 0.5B to 72B parameters.
"""

import sys
import os
sys.path.append("../../")
from dataclasses import dataclass
from typing import Optional, Tuple

# Genesis-only implementation - no framework switching
import genesis
from genesis import Tensor
import genesis.nn as nn
import genesis.nn.functional as F

import time
import numpy as np
import random

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Integer seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    # Use Genesis unified RNG system
    genesis.manual_seed(seed) 

set_seed(42)

def find_multiple(n: int, k: int) -> int:
    """
    Find the smallest multiple of k that is >= n.
    
    Args:
        n: Target number
        k: Multiple to align to
    
    Returns:
        Smallest multiple of k that is >= n
    
    Example:
        find_multiple(10, 8) returns 16
        find_multiple(16, 8) returns 16
    """
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    """
    Configuration arguments for Qwen model.
    
    Attributes:
        block_size: Maximum sequence length for input
        vocab_size: Size of the vocabulary
        n_layer: Number of transformer layers
        num_attention_heads: Number of attention heads
        hidden_size: Hidden dimension size
        intermediate_size: Size of the feed-forward network intermediate layer
        n_local_heads: Number of local attention heads (deprecated)
        num_key_value_heads: Number of key-value heads for GQA
        head_dim: Dimension of each attention head
        rope_base: Base frequency for rotary position embeddings
        max_position_embeddings: Maximum position embeddings length
        norm_eps: Epsilon for layer normalization
        rope_scaling: Optional rope scaling configuration
    """
    block_size: int = 2048
    vocab_size: int = 151936
    n_layer: int = 24
    num_attention_heads: int = 14
    hidden_size: int = 896
    intermediate_size: int = 4864
    n_local_heads: int = -1
    num_key_value_heads: int = 2
    head_dim: int = 64
    rope_base: float = 1000000.
    max_position_embeddings: int = 32768
    norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        """
        Post-initialization to compute derived parameters.
        
        Sets default values for n_local_heads and intermediate_size if not specified.
        Computes head_dim from hidden_size and num_attention_heads.
        """
        if self.n_local_heads == -1:
            self.n_local_heads = self.num_attention_heads
        if self.intermediate_size is None:
            hidden_dim = 4 * self.hidden_size
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_name(cls, name: str) -> "ModelArgs":
        """
        Create ModelArgs from a predefined configuration name.
        
        Args:
            name: Model configuration name (e.g., "Qwen-0.5B", "Qwen-7B")
        
        Returns:
            ModelArgs instance with the specified configuration
        
        Note:
            Supports fuzzy matching - will find the best match based on
            substring matching with preference for longer matches.
        """
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one "best" match
        return cls(**transformer_configs[config[0]])


class QwenModel(nn.Module):
    """
    Core Qwen transformer model.
    
    Implements the main transformer architecture with embedding layer,
    transformer blocks, normalization, and language modeling head.
    
    Attributes:
        config: Model configuration
        embed_tokens: Token embedding layer
        layers: List of transformer blocks
        norm: Final RMS normalization layer
        lm_head: Language modeling head for next token prediction
    """
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #self.lm_head.weight = self.embed_tokens.weight

        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size: int, max_seq_length: int):
        """
        Initialize KV caches for efficient inference.
        
        Args:
            max_batch_size: Maximum batch size to support
            max_seq_length: Maximum sequence length to support
        
        Note:
            Only reinitializes if requested sizes exceed current cache sizes.
            Sequence length is aligned to multiple of 8 for efficiency.
        """
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

    def forward(self, idx: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the Qwen model.
        
        Args:
            idx: Input token indices of shape (batch_size, sequence_length)
            position_ids: Optional position indices for each token
        
        Returns:
            Logits tensor of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, sequence_length = idx.shape
        
        if position_ids is None:
            position_ids = genesis.arange(0, sequence_length, device=idx.device)
            position_ids = position_ids + genesis.zeros(batch_size, sequence_length, device=idx.device)
        mask = None
        
        x = self.embed_tokens(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, position_ids, position_ids, mask)
            
        x = self.norm(x)
        x = self.lm_head(x)
        
        return x

    @classmethod
    def from_name(cls, name: str):
        """
        Create QwenModel from a predefined configuration name.
        
        Args:
            name: Model configuration name
        
        Returns:
            QwenModel instance with the specified configuration
        """
        return cls(ModelArgs.from_name(name))


class QwenForCausalLM(nn.Module):
    """
    Qwen model for causal language modeling tasks.
    
    Wrapper around QwenModel that provides a simple interface
    for language modeling with support for weight sharing between
    embedding and output layers.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model = QwenModel(config)

    def forward(self, idx: Tensor, **kwargs) -> Tensor:
        """
        Forward pass for causal language modeling.
        
        Args:
            idx: Input token indices
            **kwargs: Additional arguments passed to the model
        
        Returns:
            Logits tensor for next token prediction
        """
        logits = self.model(idx, **kwargs)
        return logits

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """
        Load model weights with support for weight tying.
        
        Args:
            state_dict: Dictionary containing model weights
            strict: Whether to strictly enforce that all keys match
        
        Note:
            Automatically shares weights between embed_tokens and lm_head
            if lm_head weights are not present in the state dict.
        """
        if "model.lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            state_dict["model.lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        return super().load_state_dict(state_dict, strict)


class RotaryEmbedding(genesis.nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    Implements the rotary position embedding mechanism that encodes
    absolute position information with rotation matrices, allowing
    the model to generalize to longer sequences.
    
    Attributes:
        inv_freq: Inverse frequencies for computing rotary embeddings
        cos_cached: Precomputed cosine values
        sin_cached: Precomputed sine values
    """
    def __init__(
        self, dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000
    ) -> None:
        super().__init__()
        self.inv_freq = genesis.tensor(1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim)))
        self.max_seq_len_cached = max_position_embeddings
        t = genesis.tensor(np.arange(self.max_seq_len_cached, dtype="float32"))
        t = t.reshape(t.shape[0], 1)
        self.inv_freq = self.inv_freq.reshape(1, self.inv_freq.shape[0])
        freqs = t @ self.inv_freq
        emb = genesis.stack((freqs, freqs), dim=-1).transpose(-1, -2).reshape(freqs.shape[0], freqs.shape[1] * 2)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Get rotary embeddings for the given sequence length.
        
        Args:
            x: Input tensor (used for shape reference)
            seq_len: Sequence length to generate embeddings for
        
        Returns:
            Tuple of (cos, sin) tensors for rotary embeddings
        """
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )

class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward layers.
    
    Implements a standard transformer block with:
    - Multi-head self-attention with RoPE
    - Feed-forward network with SiLU activation
    - RMS normalization before each sub-layer
    - Residual connections
    """
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, config.norm_eps) 
        
    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        position_ids: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            input_pos: Input position indices for KV cache
            position_ids: Position indices for rotary embeddings
            mask: Optional attention mask
        
        Returns:
            Output tensor of same shape as input
        """
        h = x + self.self_attn(self.input_layernorm(x), position_ids, mask, input_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Attention(nn.Module):
    """
    Multi-head attention module with grouped-query attention (GQA).
    
    Implements scaled dot-product attention with:
    - Rotary position embeddings
    - Grouped-query attention for efficiency
    - Optional KV caching for inference
    
    Attributes:
        q_proj, k_proj, v_proj: Query, key, value projection layers
        o_proj: Output projection layer
        rotary_emb: Rotary position embedding module
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_base
        )

        self.kv_cache = None
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute multi-head attention with rotary embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_ids: Position indices for rotary embeddings
            mask: Optional attention mask
            input_pos: Optional input positions for KV cache
        
        Returns:
            Attention output of shape (batch_size, seq_len, hidden_size)
        
        Note:
            Uses grouped-query attention where key-value heads are shared
            across multiple query heads for efficiency.
        """
        bsz, seqlen, _ = x.shape

        kv_size = self.num_attention_heads * self.head_dim

        hidden_shape = (*x.shape[:-1], -1, self.head_dim)

        q = self.q_proj(x).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(x).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=k.shape[-2])
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
        batch, num_key_value_heads, slen, head_dim = k.shape
        n_rep = self.num_attention_heads // self.num_key_value_heads
        k = k[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        k = k.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        v = v[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        v = v.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, self.hidden_size)
        y = self.o_proj(y)
        
        return y


class FeedForward(nn.Module):
    """
    Feed-forward network module.
    
    Implements the position-wise feed-forward network with:
    - Gated linear unit (GLU) variant with SiLU activation
    - Two parallel projections (gate and up) followed by down projection
    """
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor of same shape as input
        
        Note:
            Uses SwiGLU activation: silu(gate_proj(x)) * up_proj(x)
        """
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))

def rotate_half(x):
    """
    Rotate half of the hidden dimensions for RoPE.
    
    Args:
        x: Input tensor with shape (..., hidden_dim)
    
    Returns:
        Tensor with the last half of dimensions negated and swapped
        with the first half
    
    Example:
        [a, b, c, d] -> [-c, -d, a, b]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return genesis.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        cos: Cosine values for rotary embedding
        sin: Sine values for rotary embedding
        position_ids: Position indices for each token
        unsqueeze_dim: Dimension to unsqueeze cos/sin tensors
    
    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied
    
    Note:
        Applies rotation: q * cos + rotate_half(q) * sin
    """
    cos = cos[position_ids.long()].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids.long()].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
