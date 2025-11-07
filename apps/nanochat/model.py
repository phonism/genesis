"""Simple Transformer-based Language Model for nanochat.

A lightweight GPT-style architecture with multi-head attention, feed-forward networks,
and causal masking for autoregressive language modeling.
"""

import sys
import os
sys.path.append("../../")

from dataclasses import dataclass
from typing import Optional

# Backend selection: set NANOCHAT_BACKEND=torch to use PyTorch
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


@dataclass
class ModelConfig:
    """Configuration for the language model.
    
    Attributes:
        vocab_size: Size of the vocabulary
        block_size: Maximum sequence length
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        dropout: Dropout probability
    """
    vocab_size: int = 32000
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer.

    Implements scaled dot-product attention with causal masking to prevent
    attending to future positions in the sequence.
    """

    def __init__(self, config: ModelConfig):
        """Initialize attention layer.

        Args:
            config: Model configuration
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout

        # Pre-compute and cache the causal mask as a buffer (not a parameter)
        # This avoids creating it every forward pass
        self.register_buffer(
            "causal_mask",
            genesis.tril(genesis.ones((config.block_size, config.block_size)))
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.shape

        # Project to queries, keys, and values
        qkv = self.qkv_proj(x)
        # Split qkv into q, k, v (each of size n_embd)
        q = qkv[:, :, :self.n_embd]
        k = qkv[:, :, self.n_embd:2*self.n_embd]
        v = qkv[:, :, 2*self.n_embd:]

        # Reshape and transpose for multi-head attention
        q = genesis.reshape(q, (B, T, self.n_head, self.head_dim))
        k = genesis.reshape(k, (B, T, self.n_head, self.head_dim))
        v = genesis.reshape(v, (B, T, self.n_head, self.head_dim))

        q = genesis.transpose(q, 1, 2)  # (B, n_head, T, head_dim)
        k = genesis.transpose(k, 1, 2)
        v = genesis.transpose(v, 1, 2)

        # Compute attention scores with scaling
        att = (q @ genesis.transpose(k, -2, -1)) * (1.0 / (self.head_dim ** 0.5))

        # Apply cached causal mask (slice for current sequence length)
        mask = self.causal_mask[:T, :T]
        att = genesis.where(mask == 0, float("-inf"), att)

        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        if self.dropout > 0:
            att = F.dropout(att, p=self.dropout)

        # Apply attention to values
        y = att @ v
        y = genesis.transpose(y, 1, 2)
        y = genesis.reshape(y, (B, T, C))

        # Output projection
        y = self.out_proj(y)
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout)

        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: ModelConfig):
        """Initialize MLP.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""

    def __init__(self, config: ModelConfig):
        """Initialize transformer block.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connections.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoChatModel(nn.Module):
    """Simple GPT-style language model for nanochat."""

    def __init__(self, config: ModelConfig):
        """Initialize model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = config.dropout

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output projection head (no weight tying, following modern LLM trends)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            idx: Input token indices of shape (B, T)
        
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        B, T = idx.shape

        pos = genesis.arange(0, T, dtype=genesis.int64, device=idx.device)
        pos = genesis.unsqueeze(pos, 0)
        pos = genesis.broadcast_to(pos, (B, T))
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

    @genesis.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
        """Generate text autoregressively.
        
        Args:
            idx: Starting token indices of shape (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated token indices of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = genesis.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                logits = genesis.where(logits < v[:, [-1]], float("-inf"), logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = genesis.multinomial(probs, num_samples=1)
            idx = genesis.cat([idx, idx_next], dim=1)
        
        return idx
