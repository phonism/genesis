"""Transformer models for Genesis framework.

This module provides transformer-based models following HuggingFace design patterns.

Architectures:
- Dense Transformer: Standard transformer with various configurations (Llama, Qwen, etc.)
- MoE Transformer: Mixture of Experts for efficient large-scale models

Components:
- Shared components (RoPE, attention utilities)
- Unified configuration system
- Pretrained model base classes
"""

# Configuration classes
from .config import (
    TransformerConfig,
    MoEConfig,
    TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
    MOE_PRETRAINED_CONFIG_ARCHIVE_MAP,
    get_transformer_config,
    get_moe_config,
)

# Shared components
from .components import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    find_multiple,
)

# MoE models
from .moe import (
    MoEAttention,
    MoEDecoderLayer,
    MoEPreTrainedModel,
    MoEModel,
    MoEForCausalLM,
    DenseFFN,
)

# Export all
__all__ = [
    # Configuration
    "TransformerConfig",
    "MoEConfig",
    "TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
    "MOE_PRETRAINED_CONFIG_ARCHIVE_MAP",
    "get_transformer_config",
    "get_moe_config",
    # Components
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "find_multiple",
    # MoE models
    "MoEAttention",
    "MoEDecoderLayer",
    "MoEPreTrainedModel",
    "MoEModel",
    "MoEForCausalLM",
    "DenseFFN",
]
