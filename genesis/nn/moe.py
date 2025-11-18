"""Mixture of Experts (MoE) implementation for Genesis framework.

This module implements DeepSeek-style MoE with fine-grained expert segmentation
and shared experts. The implementation includes:
- MoEGate: Routing mechanism with top-k expert selection
- MoEExpert: Individual expert networks based on SwiGLU
- MoELayer: Main MoE layer combining gating and experts
- Load balancing and auxiliary loss computation
"""

import math
from typing import Optional, Tuple, List
import genesis
from genesis import Tensor
import genesis.nn as nn
import genesis.nn.functional as F
from ..function import Function
from .modules import Module, Parameter
from .modules.transformer import MultiheadAttention
from .modules.normalization import RMSNorm


class MoEGate(Module):
    """
    Gating network for routing inputs to appropriate experts.
    
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
            aux_loss = (ce * scores_for_aux.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha
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
        return x
    
    @staticmethod  
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        grad_loss = None
        if ctx.requires_aux_loss:
            grad_loss = genesis.ones(1, dtype=ctx.dtype, device=grad_output.device)
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
        """Training-time forward pass with simple expert computation."""
        # Repeat inputs for each selected expert
        hidden_states_repeated = hidden_states.repeat_interleave(self.top_k, dim=0)
        output = genesis.zeros_like(hidden_states_repeated)
        
        # Compute expert outputs
        flat_indices = expert_indices.view(-1)
        for i, expert in enumerate(self.experts):
            expert_mask = (flat_indices == i)
            if expert_mask.any():
                # Use boolean indexing to maintain autograd (no .item()!)
                expert_input = hidden_states_repeated[expert_mask]
                expert_output = expert(expert_input)
                # Assign back using boolean indexing
                output[expert_mask] = expert_output
                
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
        
        # Process each expert's tokens in batch
        start_idx = 0
        for expert_id, count in enumerate(expert_counts):
            if count.item() == 0:
                continue

            end_idx = start_idx + count.item()
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
