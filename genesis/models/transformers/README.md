# MoE Transformer Models

This directory contains Mixture of Experts (MoE) Transformer implementations following HuggingFace transformers design patterns.

## Overview

The MoE Transformer is a sparse model architecture that uses multiple expert networks and a routing mechanism to process tokens efficiently. Key features include:

- **Sparse Expert Routing**: Only top-k experts are activated per token
- **Shared Experts**: Optional always-active experts for common patterns (DeepSeek-style)
- **Flexible Architecture**: Support for various MoE configurations
- **HuggingFace Compatibility**: Familiar API design following transformers patterns

## Architecture

### Core Components

1. **MoEConfig** (`moe_config.py`)
   - Configuration class for MoE models
   - Predefined configs for popular architectures (Mixtral, DeepSeek-MoE)
   - Flexible parameter validation

2. **MoEModel** (`moe_transformer.py`)
   - Core transformer model with MoE layers
   - Token embeddings + decoder layers + normalization

3. **MoEForCausalLM** (`moe_transformer.py`)
   - Complete model for causal language modeling
   - Includes language modeling head
   - Training and generation support

### Key Modules

- **MoEAttention**: Multi-head attention with grouped-query attention (GQA)
- **MoEDecoderLayer**: Transformer block with MoE feed-forward
- **DenseFFN**: Standard dense feed-forward for non-MoE layers
- **MoERotaryEmbedding**: Rotary position embeddings (RoPE)

## Quick Start

### Basic Usage

```python
import genesis
from genesis.models.transformers import MoEConfig, MoEForCausalLM

# Create a small MoE model
config = MoEConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_local_experts=8,
    num_experts_per_tok=2,
)

model = MoEForCausalLM(config)

# Forward pass
input_ids = genesis.randint(0, config.vocab_size, (2, 32))
logits = model(input_ids)
```

### Using Predefined Configurations

```python
from genesis.models.transformers import get_moe_config, MoEForCausalLM

# Load a predefined configuration
config = get_moe_config("mixtral-8x7b")
model = MoEForCausalLM.from_pretrained("mixtral-8x7b")

# Or use the shorthand
model = MoEForCausalLM.from_pretrained("moe-small")
```

### Training

```python
import genesis.optim as optim
import genesis.nn as nn

# Create model and optimizer
model = MoEForCausalLM(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    input_ids, labels = batch

    # Forward pass with loss computation
    loss, logits = model(input_ids, labels=labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Text Generation

```python
# Generate text
model.eval()
input_ids = genesis.tensor([[1, 2, 3]])  # Starting tokens

generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
)
```

## Supported Configurations

### 1. Mixtral-Style MoE

Classic sparse MoE without shared experts:

```python
config = MoEConfig(
    hidden_size=4096,
    num_hidden_layers=32,
    num_local_experts=8,
    num_experts_per_tok=2,
    num_shared_experts=None,  # No shared experts
)
```

### 2. DeepSeek-Style MoE

Fine-grained experts with shared experts:

```python
config = MoEConfig(
    hidden_size=2048,
    num_hidden_layers=28,
    num_local_experts=64,  # Many fine-grained experts
    num_experts_per_tok=6,
    num_shared_experts=2,  # Shared experts
    moe_intermediate_size=1408,  # Smaller per-expert size
    shared_expert_intermediate_size=2816,
)
```

### 3. Custom Mixed Architecture

Alternate between dense and MoE layers:

```python
config = MoEConfig(
    num_hidden_layers=12,
    use_moe_in_all_layers=False,
    moe_layer_interval=2,  # MoE every 2 layers
    first_moe_layer=1,  # Start from layer 1
)
```

## Configuration Parameters

### Model Architecture

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Vocabulary size | 32000 |
| `hidden_size` | Hidden dimension | 4096 |
| `intermediate_size` | FFN intermediate size | 14336 |
| `num_hidden_layers` | Number of layers | 32 |
| `num_attention_heads` | Number of attention heads | 32 |
| `num_key_value_heads` | KV heads for GQA | None (= num_attention_heads) |

### MoE-Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_local_experts` | Total number of experts | 8 |
| `num_experts_per_tok` | Experts activated per token | 2 |
| `num_shared_experts` | Always-active shared experts | None |
| `moe_intermediate_size` | Expert FFN size | None (= intermediate_size) |
| `router_aux_loss_coef` | Load balancing loss weight | 0.01 |
| `scoring_func` | Router scoring ("softmax"/"sigmoid") | "softmax" |
| `norm_topk_prob` | Normalize top-k probabilities | True |

### Architecture Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_moe_in_all_layers` | Use MoE in every layer | True |
| `moe_layer_interval` | Interval for MoE layers | 1 |
| `first_moe_layer` | First layer to use MoE | 0 |
| `attention_bias` | Use bias in attention | False |
| `mlp_bias` | Use bias in MLP | False |

## Predefined Configurations

The following configurations are available via `get_moe_config()`:

- **`moe-small`**: Small model for testing (768 dim, 4 experts)
- **`mixtral-8x7b`**: Mixtral-8x7B configuration (4096 dim, 8 experts)
- **`deepseek-moe-16b`**: DeepSeek-MoE-16B (2048 dim, 64 experts + 2 shared)

## Architecture Comparison

### vs. Standard Transformer

| Aspect | Standard Transformer | MoE Transformer |
|--------|---------------------|-----------------|
| FFN | Dense, all parameters active | Sparse, only top-k experts active |
| Parameters | N_layers × FFN_size | N_layers × (N_experts × Expert_size) |
| Computation | O(d × d_ff) per token | O(d × d_expert × k) per token |
| Capacity | Limited by compute | Much higher with same compute |

### MoE Variants

| Style | Experts | Shared | Routing | Best For |
|-------|---------|--------|---------|----------|
| Mixtral | 8 large | None | Top-2 | Balanced quality/efficiency |
| DeepSeek | 64 fine | 2 shared | Top-6 | Maximum capacity |
| Custom | Variable | Optional | Configurable | Specific use cases |

## Design Principles

This implementation follows HuggingFace transformers design patterns:

1. **Configuration-First**: Separate config from model definition
2. **Modular Components**: Reusable attention, FFN, layer modules
3. **Familiar API**: Similar to `transformers.AutoModel`
4. **Type Hints**: Full type annotations for better IDE support
5. **Comprehensive Docstrings**: Detailed documentation for all components

## Implementation Details

### Load Balancing

The router includes auxiliary loss for load balancing:

- **Token-level balancing**: Encourages uniform expert usage
- **Sequence-level balancing**: Balances experts across sequences (DeepSeek-style)
- Configurable via `router_aux_loss_coef` and `seq_aux`

### Efficiency Optimizations

- **Grouped-Query Attention**: Reduces KV cache size
- **Sparse Routing**: Only activates top-k experts
- **Shared Experts**: Amortizes common computations
- **Training/Inference Modes**: Different expert dispatch strategies

### Memory Management

- **KV Caching**: Efficient autoregressive generation
- **Gradient Checkpointing**: (Can be added) Reduce memory in training
- **Mixed Precision**: Compatible with AMP training

## Examples

See `examples/moe_transformer_example.py` for comprehensive usage examples:

1. Basic model creation and forward pass
2. Using predefined configurations
3. DeepSeek-style architecture
4. Mixtral-style architecture
5. Custom mixed dense/MoE layers
6. Training loop
7. Text generation

Run the examples:

```bash
python examples/moe_transformer_example.py
```

## Testing

Run the test suite:

```bash
# All MoE tests
python -m pytest tests/test_moe_transformer.py -v

# Specific test class
python -m pytest tests/test_moe_transformer.py::TestMoEConfig -v

# Specific test
python -m pytest tests/test_moe_transformer.py::TestMoEModel::test_moe_model_forward -v
```

## Performance Considerations

### Compute Efficiency

With top-k routing, computation per token is roughly:

```
Standard FFN: O(d × d_ff)
MoE FFN:      O(d × d_expert × k + routing_overhead)
```

For `k=2` and `d_expert = d_ff/4`, MoE uses ~50% of dense FFN compute while having 4x parameters.

### Memory Usage

- **Model Size**: `N_experts × Expert_size` (but only sparse activation)
- **Activation Memory**: Similar to dense model
- **KV Cache**: Same as standard transformer (attention is unchanged)

### Scalability

MoE models scale well to:
- **Many experts**: 64+ experts (DeepSeek uses 64)
- **Large models**: Tested up to billions of parameters
- **Long sequences**: Limited by attention, not MoE

## Future Enhancements

Potential improvements for future versions:

- [ ] Expert parallelism for distributed training
- [ ] Dynamic expert capacity
- [ ] Hierarchical routing
- [ ] Expert pruning/merging
- [ ] Gradient checkpointing
- [ ] Flash attention integration
- [ ] Weight loading from HuggingFace checkpoints

## References

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{genesis_moe_transformer,
  title = {MoE Transformer for Genesis Framework},
  author = {Genesis Team},
  year = {2025},
  url = {https://github.com/genesis-ai/genesis}
}
```

## License

Same as the Genesis framework license.
