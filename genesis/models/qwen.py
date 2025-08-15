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
    Set random seed for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Genesis doesn't have global manual_seed, skip for now 

set_seed(42)

def find_multiple(n: int, k: int) -> int:
    """
    Find the smallest multiple of k that is greater than or equal to n.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    """
    Model arguments.
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
        Initialize model arguments.
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
        Load model arguments from a name.
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
    Qwen model.
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
        Setup caches for the model.
        """
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

    def forward(self, idx: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the model.
        """
        batch_size, sequence_length = idx.size()
        
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
        Load model from name.
        """
        return cls(ModelArgs.from_name(name))


class QwenForCausalLM(nn.Module):
    """
    Qwen for causal language modeling.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model = QwenModel(config)

    def forward(self, idx: Tensor, **kwargs) -> Tensor:
        """
        Forward pass of the model.
        """
        logits = self.model(idx, **kwargs)
        return logits

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """
        Load state dict with weight sharing between embed_tokens and lm_head
        """
        if "model.lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            state_dict["model.lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        return super().load_state_dict(state_dict, strict)


class RotaryEmbedding(genesis.nn.Module):
    """
    Rotary embedding.
    """
    def __init__(
        self, dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000
    ) -> None:
        super().__init__()
        self.inv_freq = genesis.Tensor(1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))) 
        self.max_seq_len_cached = max_position_embeddings
        t = genesis.Tensor(np.arange(self.max_seq_len_cached, dtype="float32"))
        t = t.reshape(t.shape[0], 1)
        self.inv_freq = self.inv_freq.reshape(1, self.inv_freq.shape[0])
        freqs = t @ self.inv_freq
        emb = genesis.stack((freqs, freqs), dim=-1).transpose(-1, -2).reshape(freqs.shape[0], freqs.shape[1] * 2)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the rotary embedding layer.
        """
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )

class TransformerBlock(nn.Module):
    """
    Transformer block.
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
        Forward pass of the transformer block.
        """
        h = x + self.self_attn(self.input_layernorm(x), position_ids, mask, input_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Attention(nn.Module):
    """
    Attention module.
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
        Forward pass of the attention module.
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
    Feed forward module.
    """
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the feed forward module.
        """
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))

def rotate_half(x):
    """
    Rotate the last half of the tensor.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return genesis.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply rotary position embedding - Genesis version.
    """
    cos = cos[position_ids.long()].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids.long()].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
