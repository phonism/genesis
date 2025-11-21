"""Dense Transformer implementation supporting both Genesis and PyTorch backends.

This module implements standard dense transformer architectures with support for:
- Rotary position embeddings (RoPE)
- Grouped-query attention (GQA)
- SwiGLU feed-forward networks
- Backend-agnostic design (works with both Genesis and PyTorch)
- Various model sizes and configurations (Llama, Qwen, etc.)

The implementation follows HuggingFace transformers design patterns and uses
shared components from the components module.

Backend Selection:
    Set NANOCHAT_BACKEND=torch to use PyTorch, otherwise uses Genesis.
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
    import torch.nn.functional as F
else:
    import genesis
    from genesis import Tensor
    import genesis.nn as nn
    import genesis.nn.functional as F

from .config import TransformerConfig, get_transformer_config
from .components import RotaryEmbedding, apply_rotary_pos_emb, rotate_half, find_multiple

# Backward compatibility alias
ModelArgs = TransformerConfig


class QwenModel(nn.Module):
    """Core Qwen transformer model.
    
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
        """Initialize KV caches for efficient inference.
        
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
        """Forward pass through the Qwen model.

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
        """Create QwenModel from a predefined configuration name.
        
        Args:
            name: Model configuration name
        
        Returns:
            QwenModel instance with the specified configuration
        """
        return cls(ModelArgs.from_name(name))


class QwenForCausalLM(nn.Module):
    """Qwen model for causal language modeling tasks.
    
    Wrapper around QwenModel that provides a simple interface
    for language modeling with support for weight sharing between
    embedding and output layers.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model = QwenModel(config)

    def forward(self, idx: Tensor, **kwargs) -> Tensor:
        """Forward pass for causal language modeling.
        
        Args:
            idx: Input token indices
            **kwargs: Additional arguments passed to the model
        
        Returns:
            Logits tensor for next token prediction
        """
        logits = self.model(idx, **kwargs)
        return logits

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load model weights with support for weight tying.
        
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


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers.
    
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
        """Forward pass through the transformer block.
        
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
    """Multi-head attention module with grouped-query attention (GQA).
    
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
        """Compute multi-head attention with rotary embeddings.
        
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
    """Feed-forward network module.
    
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
        """Apply feed-forward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor of same shape as input
        
        Note:
            Uses SwiGLU activation: silu(gate_proj(x)) * up_proj(x)
        """
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))
