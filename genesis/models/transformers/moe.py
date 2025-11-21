"""Mixture of Experts (MoE) Transformer implementation.

This module implements a complete MoE Transformer model following HuggingFace transformers
design patterns. It supports various MoE architectures including DeepSeek-MoE and Mixtral-style
sparse expert routing.

The implementation includes:
- MoEGate: Routing mechanism with top-k expert selection
- MoEExpert: Individual expert networks based on SwiGLU
- MoELayer: Main MoE layer combining gating and experts
- AddAuxiliaryLoss: Auxiliary loss for load balancing
- MoETransformerBlock: Transformer block with MoE feed-forward
- MoEAttention: Multi-head attention with grouped-query attention (GQA)
- MoEDecoderLayer: Transformer decoder layer with MoE feed-forward
- MoEPreTrainedModel: Base class for MoE models
- MoEModel: Core MoE transformer model
- MoEForCausalLM: MoE model for causal language modeling
"""

import math
from typing import Optional, Tuple, List, Union
import numpy as np

import genesis
from genesis import Tensor
from genesis.function import Function
import genesis.nn as nn
import genesis.nn.functional as F
from genesis.nn.modules import Module, Parameter
from genesis.nn.modules.transformer import MultiheadAttention
from genesis.nn.modules.normalization import RMSNorm
from .config import MoEConfig
from .components import RotaryEmbedding, apply_rotary_pos_emb, rotate_half


# ============================================================================
# Core MoE Components
# ============================================================================

class MoEGate(Module):
    """Gating network for routing inputs to appropriate experts.

    Implements the routing mechanism from DeepSeek MoE with:
    - Learnable routing weights
    - Top-k expert selection with softmax scoring
    - Normalization of top-k probabilities
    - Auxiliary loss for load balancing
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        scoring_func: str = "softmax"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func

        # Learnable routing weights
        self.weight = Parameter(genesis.randn(num_experts, hidden_size, std=0.02))

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Route inputs to top-k experts.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of (expert_indices, expert_weights, auxiliary_loss)
            - expert_indices: Selected expert indices (batch_size * seq_len, top_k)
            - expert_weights: Expert weights (batch_size * seq_len, top_k)
            - auxiliary_loss: Load balancing loss (scalar) or None
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Flatten for routing computation
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # (B*L, H)

        # Compute routing scores
        logits = hidden_states_flat @ self.weight.transpose(0, 1)  # (B*L, E)

        if self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            raise NotImplementedError(f"Unsupported scoring function: {self.scoring_func}")

        # Select top-k experts
        topk_weights, topk_indices = genesis.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # Normalize top-k probabilities to sum to 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        # Compute auxiliary loss for load balancing during training
        aux_loss = None
        if self.training and self.aux_loss_alpha > 0.0:
            aux_loss = self._compute_auxiliary_loss(
                scores, topk_indices, batch_size, seq_len
            )

        return topk_indices, topk_weights, aux_loss

    def _compute_auxiliary_loss(
        self,
        scores: Tensor,
        topk_indices: Tensor,
        batch_size: int,
        seq_len: int
    ) -> Tensor:
        """
        Compute auxiliary loss for expert load balancing.

        The auxiliary loss encourages balanced expert utilization by penalizing
        scenarios where some experts are used much more frequently than others.
        """
        if self.seq_aux:
            # Sequence-level auxiliary loss
            scores_for_aux = scores.view(batch_size, seq_len, -1)
            topk_indices_for_aux = topk_indices.view(batch_size, -1)

            # Compute expert usage counts
            ce = genesis.zeros(batch_size, self.num_experts, device=scores.device)
            ones = genesis.ones(batch_size, seq_len * self.top_k, device=scores.device)
            ce = ce.scatter_add(1, topk_indices_for_aux, ones)
            ce = ce / (seq_len * self.top_k / self.num_experts)

            # Auxiliary loss: ce * mean_scores
            mean_scores = scores_for_aux.mean(dim=1)
            aux_loss = (ce * mean_scores).sum(dim=1).mean() * self.aux_loss_alpha
        else:
            # Token-level auxiliary loss (original Switch Transformer style)
            mask_ce = F.one_hot(topk_indices.view(-1), num_classes=self.num_experts)
            ce = mask_ce.float().mean(0)  # Expert usage frequency
            pi = scores.mean(0)  # Expert assignment probability
            fi = ce * self.num_experts  # Fraction of tokens assigned to each expert
            aux_loss = (pi * fi).sum() * self.aux_loss_alpha

        return aux_loss


class MoEExpert(Module):
    """
    Individual expert network based on SwiGLU feed-forward architecture.

    Each expert is a standard feed-forward network with:
    - Gate projection (for SwiGLU activation)
    - Up projection (linear transformation)
    - Down projection (output transformation)
    - SiLU activation function
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU components
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.silu = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply expert transformation using SwiGLU activation.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # SwiGLU: silu(gate_proj(x)) * up_proj(x)
        gate_output = self.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        return self.down_proj(intermediate)


class AddAuxiliaryLoss(Function):
    """
    Custom autograd function to add auxiliary loss during backpropagation.

    This allows the auxiliary loss to contribute to gradients without affecting
    the forward pass output.
    """

    @staticmethod
    def forward(ctx, x: Tensor, loss: Tensor) -> Tensor:
        assert loss.numel() == 1, "Auxiliary loss must be a scalar"
        ctx.dtype = loss.dtype
        ctx.requires_aux_loss = loss.requires_grad
        # IMPORTANT: Must create a new tensor, not return input directly,
        # otherwise the computational graph won't be properly connected
        return x.view(*x.shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        grad_loss = None
        if ctx.requires_aux_loss:
            # Return scalar gradient for auxiliary loss (not shape (1,))
            grad_loss = genesis.tensor(1.0, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoELayer(Module):
    """
    Main Mixture of Experts layer.

    Combines gating network with expert networks and optional shared experts.
    Implements efficient sparse computation where only top-k experts are activated
    per token, plus always-active shared experts for common patterns.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        num_shared_experts: Optional[int] = None,
        shared_intermediate_size: Optional[int] = None,
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        expert_bias: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts

        # Gating network
        self.gate = MoEGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_alpha=aux_loss_alpha,
            seq_aux=seq_aux,
            norm_topk_prob=norm_topk_prob
        )

        # Routed experts
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size, bias=expert_bias)
            for _ in range(num_experts)
        ])

        # Shared experts (always active)
        if num_shared_experts is not None and num_shared_experts > 0:
            if shared_intermediate_size is None:
                shared_intermediate_size = intermediate_size * num_shared_experts
            self.shared_experts = MoEExpert(
                hidden_size, shared_intermediate_size, bias=expert_bias
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of same shape as input
        """
        identity = hidden_states
        orig_shape = hidden_states.shape

        # Route to experts
        expert_indices, expert_weights, aux_loss = self.gate(hidden_states)

        # Flatten for expert computation
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_weights.view(-1, 1)

        # Compute expert outputs
        if self.training:
            # Training mode: use simple but memory-intensive approach
            output = self._training_forward(
                hidden_states_flat, flat_expert_indices, expert_weights
            )
        else:
            # Inference mode: use memory-efficient approach
            output = self._inference_forward(
                hidden_states_flat, flat_expert_indices, flat_expert_weights
            )

        output = output.view(*orig_shape)

        # Add auxiliary loss during training
        if aux_loss is not None:
            output = AddAuxiliaryLoss.apply(output, aux_loss)

        # Add shared expert output if present
        if self.shared_experts is not None:
            shared_output = self.shared_experts(identity)
            output = output + shared_output

        return output

    def _training_forward(
        self,
        hidden_states: Tensor,
        expert_indices: Tensor,
        expert_weights: Tensor
    ) -> Tensor:
        """Training-time forward pass with simple expert computation.

        Uses scatter to avoid in-place operations for proper gradient flow.
        """
        # Repeat inputs for each selected expert
        hidden_states_repeated = hidden_states.repeat_interleave(self.top_k, dim=0)

        # Simplified approach: concatenate all expert outputs in expert order,
        # then use argsort to reorder
        flat_indices = expert_indices.view(-1)

        # Collect outputs from all experts in expert order
        expert_outputs_list = []
        for i, expert in enumerate(self.experts):
            expert_mask = (flat_indices == i)
            expert_input = hidden_states_repeated[expert_mask]
            if expert_input.shape[0] > 0:
                expert_output = expert(expert_input)
                expert_outputs_list.append(expert_output)

        # Concatenate all expert outputs
        if len(expert_outputs_list) > 0:
            concat_outputs = genesis.cat(expert_outputs_list, dim=0)

            # Create mapping from flat_indices to output positions
            # Sort flat_indices to group by expert, then unsort to get back to token order
            sorted_indices_idx = flat_indices.argsort()  # Indices that would sort flat_indices
            unsort_indices_idx = sorted_indices_idx.argsort()  # Indices to unsort

            # Apply unsort to get outputs in original token order
            output = concat_outputs[unsort_indices_idx]
        else:
            output = genesis.zeros_like(hidden_states_repeated)

        # Weight and combine expert outputs
        output = output.view(*expert_weights.shape, -1)  # (batch*seq, top_k, hidden)
        output = (output * expert_weights.unsqueeze(-1)).sum(dim=1)

        return output

    def _inference_forward(
        self,
        hidden_states: Tensor,
        expert_indices: Tensor,
        expert_weights: Tensor
    ) -> Tensor:
        """Inference-time forward pass with memory-efficient expert computation."""
        expert_cache = genesis.zeros_like(hidden_states)

        # Sort by expert index for efficient batching
        sorted_indices = expert_indices.argsort()
        expert_counts = genesis.bincount(expert_indices, minlength=self.num_experts)

        # OPTIMIZATION: Convert counts to CPU once to avoid multiple .item() calls
        expert_counts_cpu = expert_counts.detach().numpy()

        # Process each expert's tokens in batch
        start_idx = 0
        for expert_id, count in enumerate(expert_counts_cpu):
            if count == 0:
                continue

            end_idx = start_idx + int(count)
            token_indices = sorted_indices[start_idx:end_idx] // self.top_k

            # Get tokens for this expert
            expert_tokens = hidden_states[token_indices]
            expert_output = self.experts[expert_id](expert_tokens)

            # Apply weights and accumulate
            weights = expert_weights[sorted_indices[start_idx:end_idx]]
            weighted_output = expert_output * weights

            # Scatter results back to original positions
            expert_cache[token_indices] += weighted_output

            start_idx = end_idx

        return expert_cache


class MoETransformerBlock(Module):
    """
    Transformer block with MoE feed-forward layer.

    Replaces the standard feed-forward layer with a MoE layer while keeping
    the attention mechanism and layer normalization unchanged.
    """

    def __init__(self, config):
        super().__init__()

        self.self_attn = MultiheadAttention(config)

        # MoE layer instead of standard feed-forward
        self.mlp = MoELayer(
            hidden_size=config.hidden_size,
            num_experts=getattr(config, 'num_experts', 8),
            top_k=getattr(config, 'top_k', 2),
            intermediate_size=getattr(config, 'moe_intermediate_size', config.intermediate_size),
            num_shared_experts=getattr(config, 'num_shared_experts', None),
            aux_loss_alpha=getattr(config, 'aux_loss_alpha', 0.01),
            seq_aux=getattr(config, 'seq_aux', True),
            norm_topk_prob=getattr(config, 'norm_topk_prob', True),
            expert_bias=getattr(config, 'expert_bias', False)
        )

        self.input_layernorm = RMSNorm(config.hidden_size, getattr(config, 'norm_eps', 1e-6))
        self.post_attention_layernorm = RMSNorm(config.hidden_size, getattr(config, 'norm_eps', 1e-6))

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        position_ids: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass through MoE transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            input_pos: Input position indices for KV cache
            position_ids: Position indices for rotary embeddings
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        h = x + self.self_attn(self.input_layernorm(x), position_ids, mask, input_pos)

        # MoE feed-forward with residual connection
        out = h + self.mlp(self.post_attention_layernorm(h))

        return out


# ============================================================================
# Full MoE Transformer Models
# ============================================================================


class MoEAttention(nn.Module):
    """
    Multi-head attention module with grouped-query attention (GQA) for MoE Transformer.

    Implements scaled dot-product attention with:
    - Rotary position embeddings (RoPE)
    - Grouped-query attention for efficiency
    - Optional KV caching for autoregressive generation
    - Support for causal masking

    Args:
        config: MoE model configuration.
        layer_idx: Index of this layer in the model (for KV cache).

    Attributes:
        hidden_size: Dimension of hidden states.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key-value heads (for GQA).
        head_dim: Dimension of each attention head.
        q_proj, k_proj, v_proj: Query, key, value projection layers.
        o_proj: Output projection layer.
        rotary_emb: Rotary position embedding module.
    """

    def __init__(self, config: MoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = 0.0  # Could be added to config

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        # Projection layers
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # KV cache (set externally if needed)
        self.kv_cache = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass of the attention module.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len).
            position_ids: Optional position indices for RoPE.
            past_key_value: Optional cached (key, value) from previous forward pass.
            use_cache: Whether to return cached key-value states.

        Returns:
            Tuple of (attention_output, new_past_key_value).
            - attention_output: shape (batch_size, seq_len, hidden_size)
            - new_past_key_value: cached (key, value) if use_cache else None
        """
        bsz, seq_len, _ = hidden_states.shape

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = genesis.arange(0, seq_len, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, seq_len)

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)

        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        key_states = key_states.transpose(1, 2)

        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[-2])
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Handle past key-value cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = genesis.cat((past_key, key_states), dim=2)
            value_states = genesis.cat((past_value, value_states), dim=2)

        # Update cache if requested
        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat K, V for grouped-query attention
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = self._repeat_kv(key_states, self.num_attention_heads // self.num_key_value_heads)
            value_states = self._repeat_kv(value_states, self.num_attention_heads // self.num_key_value_heads)

        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=(attention_mask is None and seq_len > 1),
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _repeat_kv(self, hidden_states: Tensor, n_rep: int) -> Tensor:
        """
        Repeat key/value heads for grouped-query attention.

        Args:
            hidden_states: Tensor of shape (batch, num_kv_heads, slen, head_dim).
            n_rep: Number of times to repeat each head.

        Returns:
            Tensor of shape (batch, num_kv_heads * n_rep, slen, head_dim).
        """
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states

        # Expand and reshape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class MoEDecoderLayer(nn.Module):
    """
    Transformer decoder layer with MoE feed-forward network.

    This layer implements a standard transformer decoder block with:
    - Multi-head self-attention
    - MoE feed-forward network (or dense FFN for non-MoE layers)
    - RMS normalization before each sub-layer
    - Residual connections

    Args:
        config: MoE model configuration.
        layer_idx: Index of this layer in the model.

    Attributes:
        self_attn: Multi-head attention module.
        mlp: MoE or dense feed-forward network.
        input_layernorm: RMS normalization before attention.
        post_attention_layernorm: RMS normalization before FFN.
    """

    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Self-attention
        self.self_attn = MoEAttention(config, layer_idx)

        # Determine whether to use MoE or dense FFN for this layer
        self.is_moe_layer = self._is_moe_layer(layer_idx, config)

        if self.is_moe_layer:
            # MoE feed-forward network
            self.mlp = MoELayer(
                hidden_size=config.hidden_size,
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                intermediate_size=config.moe_intermediate_size,
                num_shared_experts=config.num_shared_experts,
                shared_intermediate_size=config.shared_expert_intermediate_size,
                aux_loss_alpha=config.router_aux_loss_coef,
                seq_aux=config.seq_aux,
                norm_topk_prob=config.norm_topk_prob,
                expert_bias=config.expert_bias,
            )
        else:
            # Dense feed-forward network (standard Transformer FFN)
            self.mlp = self._build_dense_mlp(config)

        # Layer normalization
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def _is_moe_layer(self, layer_idx: int, config: MoEConfig) -> bool:
        """
        Determine whether this layer should use MoE.

        Args:
            layer_idx: Index of this layer.
            config: Model configuration.

        Returns:
            True if this layer should use MoE, False otherwise.
        """
        if config.use_moe_in_all_layers:
            return True

        # Use MoE every moe_layer_interval layers, starting from first_moe_layer
        if layer_idx < config.first_moe_layer:
            return False

        return (layer_idx - config.first_moe_layer) % config.moe_layer_interval == 0

    def _build_dense_mlp(self, config: MoEConfig) -> nn.Module:
        """
        Build a dense (non-MoE) feed-forward network.

        Args:
            config: Model configuration.

        Returns:
            Dense FFN module.
        """
        return DenseFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            bias=config.mlp_bias,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position indices for RoPE.
            past_key_value: Optional cached key-value states.
            use_cache: Whether to return cached key-value states.

        Returns:
            Tuple of (layer_output, new_past_key_value).
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class DenseFFN(nn.Module):
    """
    Dense feed-forward network (standard Transformer FFN).

    Implements the position-wise feed-forward network with:
    - Gated linear unit (GLU) variant with configurable activation
    - Two parallel projections (gate and up) followed by down projection

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Intermediate dimension.
        hidden_act: Activation function ("silu", "gelu", "relu").
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        # Activation function
        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif hidden_act == "relu":
            self.act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {hidden_act}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward transformation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of same shape as input.

        Note:
            Uses gated activation: act_fn(gate_proj(x)) * up_proj(x)
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoEPreTrainedModel(nn.Module):
    """
    Base class for MoE models, providing weight initialization.

    This class provides common functionality for MoE models, including:
    - Weight initialization following the specified initialization range
    - Common model configuration handling
    - apply() method for recursive module operations

    Args:
        config: MoE model configuration.
    """

    config_class = MoEConfig

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

    def apply(self, fn):
        """
        Apply a function recursively to every submodule (including self).

        This method mimics PyTorch's Module.apply() behavior, allowing
        recursive application of initialization or other operations.

        Args:
            fn: Function to apply to each module.

        Returns:
            self for method chaining.

        Example:
            ```python
            def init_weights(module):
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)

            model.apply(init_weights)
            ```
        """
        for module in self.modules():
            fn(module)
        return self

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights for different module types.

        This method is called by apply() during model initialization.
        It handles initialization for:
        - Linear layers: Normal distribution with configurable std
        - Embedding layers: Normal distribution with configurable std
        - Other layers: No special initialization (use module defaults)

        Args:
            module: Module to initialize.

        Note:
            For Linear and Embedding layers, this will override the default
            initialization done in the module's __init__. For other modules,
            it leaves the initialization as-is.
        """
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            # Initialize linear layer weights with normal distribution
            # Check if weight has 'data' attribute or is a Parameter
            if hasattr(module, 'weight') and module.weight is not None:
                # Reinitialize weight
                new_weight = genesis.randn(*module.weight.shape, std=std, device=module.weight.device)
                # Preserve the Parameter wrapper
                module.weight.storage = new_weight.storage
                module.weight._shape = new_weight.shape
                module.weight._stride = new_weight.stride
                module.weight._offset = new_weight.offset

            if hasattr(module, 'bias') and module.bias is not None:
                # Reinitialize bias to zeros
                new_bias = genesis.zeros(*module.bias.shape, device=module.bias.device)
                module.bias.storage = new_bias.storage
                module.bias._shape = new_bias.shape
                module.bias._stride = new_bias.stride
                module.bias._offset = new_bias.offset

        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights with normal distribution
            if hasattr(module, 'weight') and module.weight is not None:
                new_weight = genesis.randn(*module.weight.shape, std=std, device=module.weight.device)
                module.weight.storage = new_weight.storage
                module.weight._shape = new_weight.shape
                module.weight._stride = new_weight.stride
                module.weight._offset = new_weight.offset


class MoEModel(MoEPreTrainedModel):
    """
    Core MoE Transformer model.

    This is the main transformer model with MoE layers. It consists of:
    - Token embeddings
    - Stack of MoE decoder layers
    - Final layer normalization

    The model outputs hidden states without a task-specific head.

    Args:
        config: MoE model configuration.

    Attributes:
        embed_tokens: Token embedding layer.
        layers: List of MoE decoder layers.
        norm: Final RMS normalization layer.
    """

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [MoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        # Initialize weights with custom initialization strategy
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
        """
        Forward pass through the MoE model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
            position_ids: Optional position IDs of shape (batch_size, seq_len).
            past_key_values: Optional list of cached key-value states from previous forward passes.
            use_cache: Whether to return cached key-value states.

        Returns:
            Tuple of (hidden_states, new_past_key_values).
            - hidden_states: shape (batch_size, seq_len, hidden_size)
            - new_past_key_values: list of cached (key, value) tuples if use_cache else None
        """
        batch_size, seq_len = input_ids.shape

        if use_cache is None:
            use_cache = self.config.use_cache

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = genesis.arange(0, seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Initialize past_key_values if needed
        if past_key_values is None and use_cache:
            past_key_values = [None] * len(self.layers)
        elif not use_cache:
            past_key_values = [None] * len(self.layers)

        # Process through transformer layers
        new_past_key_values = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            hidden_states, new_past_key_value = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                new_past_key_values.append(new_past_key_value)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_past_key_values


class MoEForCausalLM(MoEPreTrainedModel):
    """
    MoE Transformer model for causal language modeling.

    This model extends MoEModel with a language modeling head for next-token prediction.
    It's suitable for tasks like text generation and language modeling.

    Args:
        config: MoE model configuration.

    Attributes:
        model: Core MoE transformer model.
        lm_head: Language modeling head (linear layer).
    """

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.model = MoEModel(config)
        self.vocab_size = config.vocab_size

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for causal language modeling.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            past_key_values: Optional cached key-value states.
            use_cache: Whether to return cached key-value states.
            labels: Optional labels for computing language modeling loss.

        Returns:
            If labels is None: logits of shape (batch_size, seq_len, vocab_size)
            If labels is provided: tuple of (loss, logits)
        """
        # Forward through model
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)

            loss = loss_fct(shift_logits, shift_labels)
            return loss, logits

        return logits

    @classmethod
    def from_pretrained(cls, config_or_path: Union[str, MoEConfig]) -> "MoEForCausalLM":
        """
        Load a pretrained MoE model.

        Args:
            config_or_path: Either a MoEConfig instance or a string path/name.

        Returns:
            MoEForCausalLM instance.

        Note:
            Currently only supports loading from config. Checkpoint loading
            would need to be implemented separately.
        """
        if isinstance(config_or_path, str):
            from .config import get_moe_config
            config = get_moe_config(config_or_path)
        else:
            config = config_or_path

        return cls(config)

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: If set, only sample from top k tokens.
            top_p: If set, use nucleus sampling with this probability mass.

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens).
        """
        self.eval()  # Set to evaluation mode
        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self(generated)

            # Get logits for last token
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = genesis.topk(next_token_logits, k=top_k, dim=-1)
                # Set other logits to -inf
                next_token_logits = genesis.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = genesis.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = genesis.cat((generated, next_token), dim=1)

        return generated
