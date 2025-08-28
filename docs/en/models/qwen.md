# Qwen Model Implementation

## Overview

The Genesis framework includes a built-in implementation of the Qwen (Tongyi Qianwen) large language model, supporting complete training and inference workflows.

## Model Architecture

The Qwen model is based on the Transformer architecture with the following features:

- **Attention Mechanism**: Multi-Head Attention with RoPE (Rotary Position Embedding)
- **Activation Function**: SwiGLU activation
- **Layer Normalization**: RMSNorm
- **Position Encoding**: Rotary Position Embedding (RoPE)

## Quick Start

### Basic Inference

```python
import genesis
from genesis.models.qwen import QwenModel, QwenConfig

# Create model configuration
config = QwenConfig(
    vocab_size=32000,
    n_layer=24,
    n_head=16,
    n_embd=2048,
    max_seq_len=2048
)

# Create model
model = QwenModel(config)

# Inference
input_ids = genesis.tensor([[1, 2, 3, 4, 5]])  # [batch_size, seq_len]
output = model(input_ids)
print(f"Output shape: {output.shape}")  # [1, 5, 32000]
```

### Training Example

```python
import genesis.optim as optim
import genesis.nn as nn

# Create optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Training loop
for batch in dataloader:
    input_ids, labels = batch
    
    # Forward pass
    logits = model(input_ids)
    
    # Calculate loss
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Parameter update
    optimizer.step()
```

## Configuration Parameters

### QwenConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 32000 | Vocabulary size |
| `n_layer` | int | 24 | Number of Transformer layers |
| `n_head` | int | 16 | Number of attention heads |
| `n_embd` | int | 2048 | Hidden dimension |
| `max_seq_len` | int | 2048 | Maximum sequence length |
| `dropout` | float | 0.1 | Dropout probability |
| `bias` | bool | False | Whether to use bias |

## Performance Optimization

### Mixed Precision Training

```python
# Enable mixed precision
genesis.enable_autocast = True

with genesis.autocast():
    logits = model(input_ids)
    loss = criterion(logits, labels)

# Gradient scaling
scaler = genesis.GradScaler()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Checkpointing

```python
# Enable gradient checkpointing to save memory
model.gradient_checkpointing = True
```

## Application Examples

### Text Generation

```python
def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt)
    input_tensor = genesis.tensor([input_ids])
    
    with genesis.no_grad():
        for _ in range(max_length):
            logits = model(input_tensor)
            next_token = logits[0, -1].argmax()
            input_tensor = genesis.cat([input_tensor, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_tensor[0].tolist())

# Usage example
generated = generate_text(model, tokenizer, "Today's weather")
print(generated)
```

### Fine-tuning Training

Refer to `apps/llm/train_sft_qwen.py` for complete SFT (Supervised Fine-tuning) implementation.

## File Structure

- `genesis/models/qwen.py` - Model implementation
- `apps/llm/qwen_model.py` - Training configuration and utilities
- `apps/llm/train_sft_qwen.py` - SFT training script
- `apps/llm/chat_qwen.py` - Inference chat script

## Related Resources

- [Qwen Official Paper](https://arxiv.org/abs/2309.16609)
- [Mixed Precision Training Guide](../tutorials/mixed-precision.md)
- [LLM Training Hands-on](../tutorials/llm-training.md)