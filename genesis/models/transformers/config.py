"""Configuration classes for Transformer models.

This module provides configuration classes for different transformer architectures.
All configurations follow a consistent pattern and can be easily converted to/from
dictionaries for serialization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class TransformerConfig:
    """
    Base configuration class for dense Transformer models.

    This configuration supports standard transformer architectures with various
    features like grouped-query attention (GQA), rotary position embeddings (RoPE),
    and SwiGLU activation.

    Args:
        vocab_size: Size of the vocabulary. Defaults to 32000.
        hidden_size: Dimension of the hidden representations. Defaults to 4096.
        intermediate_size: Dimension of the feed-forward network. Defaults to 14336.
        num_hidden_layers: Number of transformer layers. Defaults to 32.
        num_attention_heads: Number of attention heads. Defaults to 32.
        num_key_value_heads: Number of key-value heads for GQA. If None, equals num_attention_heads.
        head_dim: Dimension of each attention head. If None, computed as hidden_size // num_attention_heads.
        max_position_embeddings: Maximum sequence length. Defaults to 4096.
        rope_theta: Base period for RoPE embeddings. Defaults to 10000.0.
        rope_scaling: Optional rope scaling configuration. Defaults to None.
        norm_eps: Epsilon for layer normalization. Defaults to 1e-6.
        use_cache: Whether to use KV cache during generation. Defaults to True.
        tie_word_embeddings: Whether to tie input and output embeddings. Defaults to False.
        attention_bias: Whether to use bias in attention layers. Defaults to False.
        mlp_bias: Whether to use bias in MLP layers. Defaults to False.
        hidden_act: Activation function for FFN ("silu", "gelu", "relu"). Defaults to "silu".
        initializer_range: Standard deviation for weight initialization. Defaults to 0.02.

    Example:
        ```python
        # Standard transformer configuration
        config = TransformerConfig(
            vocab_size=50000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12
        )

        # Llama-style configuration with GQA
        config = TransformerConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            rope_theta=10000.0
        )
        ```
    """

    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None

    # Position embeddings
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # Normalization and regularization
    norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False

    # Layer options
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        """
        Post-initialization to compute derived parameters and validate configuration.
        """
        # Set default num_key_value_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Compute head_dim
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """
        Validate the configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

    @property
    def n_layer(self) -> int:
        """Backward compatibility alias for num_hidden_layers."""
        return self.num_hidden_layers

    @property
    def rope_base(self) -> float:
        """Backward compatibility alias for rope_theta."""
        return self.rope_theta

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TransformerConfig":
        """
        Create a configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            TransformerConfig instance.
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "norm_eps": self.norm_eps,
            "use_cache": self.use_cache,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
            "hidden_act": self.hidden_act,
            "initializer_range": self.initializer_range,
        }


@dataclass
class MoEConfig(TransformerConfig):
    """
    Configuration class for Mixture of Experts (MoE) Transformer models.

    Extends TransformerConfig with MoE-specific parameters. Supports various
    MoE architectures including DeepSeek-MoE and Mixtral-style routing.

    Args:
        (Inherits all TransformerConfig args, plus:)

        # MoE specific parameters
        num_experts_per_tok: Number of experts to route each token to (top-k). Defaults to 2.
        num_local_experts: Total number of experts in each MoE layer. Defaults to 8.
        num_shared_experts: Number of shared experts (always active). None means no shared experts.
        moe_intermediate_size: Intermediate size for each expert. If None, uses intermediate_size.
        shared_expert_intermediate_size: Intermediate size for shared experts.
        router_aux_loss_coef: Weight for the auxiliary load balancing loss. Defaults to 0.01.
        router_z_loss_coef: Weight for the router z-loss. Defaults to 0.001.
        scoring_func: Scoring function for the router ("softmax" or "sigmoid"). Defaults to "softmax".
        seq_aux: Whether to use sequence-level auxiliary loss. Defaults to True.
        norm_topk_prob: Whether to normalize top-k probabilities. Defaults to True.
        expert_bias: Whether to use bias in expert layers. Defaults to False.

        # Architecture options
        use_moe_in_all_layers: Whether to use MoE in all layers or alternate with dense FFN.
        moe_layer_interval: If not using MoE in all layers, interval for MoE layers.
        first_moe_layer: First layer to use MoE (0-indexed). Defaults to 0.

    Example:
        ```python
        # DeepSeek-MoE style
        config = MoEConfig(
            hidden_size=4096,
            num_local_experts=64,
            num_experts_per_tok=6,
            num_shared_experts=2
        )

        # Mixtral style
        config = MoEConfig(
            hidden_size=4096,
            num_local_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=None
        )
        ```
    """

    # MoE specific
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    num_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    scoring_func: str = "softmax"
    seq_aux: bool = True
    norm_topk_prob: bool = True
    expert_bias: bool = False

    # Architecture options
    use_moe_in_all_layers: bool = True
    moe_layer_interval: int = 1
    first_moe_layer: int = 0

    def __post_init__(self):
        """
        Post-initialization to compute derived parameters and validate configuration.
        """
        # Call parent post_init
        super().__post_init__()

        # Set default moe_intermediate_size
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size

        # Validate MoE-specific configuration
        self._validate_moe_config()

    def _validate_moe_config(self):
        """
        Validate MoE-specific configuration parameters.

        Raises:
            ValueError: If any MoE parameter is invalid.
        """
        if self.num_experts_per_tok > self.num_local_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) cannot be greater than "
                f"num_local_experts ({self.num_local_experts})"
            )

        if self.scoring_func not in ["softmax", "sigmoid"]:
            raise ValueError(
                f"scoring_func must be 'softmax' or 'sigmoid', got '{self.scoring_func}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        base_dict = super().to_dict()
        moe_dict = {
            "num_experts_per_tok": self.num_experts_per_tok,
            "num_local_experts": self.num_local_experts,
            "num_shared_experts": self.num_shared_experts,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "router_z_loss_coef": self.router_z_loss_coef,
            "scoring_func": self.scoring_func,
            "seq_aux": self.seq_aux,
            "norm_topk_prob": self.norm_topk_prob,
            "expert_bias": self.expert_bias,
            "use_moe_in_all_layers": self.use_moe_in_all_layers,
            "moe_layer_interval": self.moe_layer_interval,
            "first_moe_layer": self.first_moe_layer,
        }
        return {**base_dict, **moe_dict}


# Predefined configurations
TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llama-7b": {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
    },
    "llama-13b": {
        "vocab_size": 32000,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 40,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
    },
    "qwen-0.5b": {
        "vocab_size": 151936,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
    },
}

MOE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "deepseek-moe-16b": {
        "hidden_size": 2048,
        "intermediate_size": 10944,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_experts_per_tok": 6,
        "num_local_experts": 64,
        "num_shared_experts": 2,
        "moe_intermediate_size": 1408,
        "shared_expert_intermediate_size": 2816,
        "vocab_size": 102400,
        "max_position_embeddings": 4096,
    },
    "mixtral-8x7b": {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_experts_per_tok": 2,
        "num_local_experts": 8,
        "num_shared_experts": None,
        "vocab_size": 32000,
        "max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
    },
    "moe-small": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "num_experts_per_tok": 2,
        "num_local_experts": 4,
        "num_shared_experts": None,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
    },
}


def get_transformer_config(model_name: str) -> TransformerConfig:
    """
    Get a predefined dense transformer configuration by name.

    Args:
        model_name: Name of the predefined configuration.

    Returns:
        TransformerConfig instance.

    Raises:
        ValueError: If the model name is not found.
    """
    if model_name not in TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP:
        available = ", ".join(TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
        raise ValueError(
            f"Unknown model name '{model_name}'. Available configurations: {available}"
        )

    return TransformerConfig.from_dict(TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name])


def get_moe_config(model_name: str) -> MoEConfig:
    """
    Get a predefined MoE configuration by name.

    Args:
        model_name: Name of the predefined configuration.

    Returns:
        MoEConfig instance.

    Raises:
        ValueError: If the model name is not found.
    """
    if model_name not in MOE_PRETRAINED_CONFIG_ARCHIVE_MAP:
        available = ", ".join(MOE_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
        raise ValueError(
            f"Unknown model name '{model_name}'. Available configurations: {available}"
        )

    return MoEConfig.from_dict(MOE_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name])
