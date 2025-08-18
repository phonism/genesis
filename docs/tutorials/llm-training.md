# Large Language Model Training Guide

This comprehensive guide covers training large language models (LLMs) using Genesis's Qwen implementation, from basic setup to advanced optimization techniques.

## Overview

Genesis provides LLM training capabilities through two approaches:
1. **Pure Genesis Implementation**: Native Qwen model in `genesis.models.qwen`
2. **Hybrid Approach**: Integration with PyTorch/Transformers for production training

This guide covers both approaches, starting with the pure Genesis implementation and then showing the production-ready hybrid approach used in `apps/llm/`.

## Prerequisites

- GPU with at least 16GB VRAM (A100/A800 recommended for large models)
- CUDA 11.8+ and appropriate drivers
- Python 3.8+ with Genesis installed
- For hybrid approach: PyTorch and Transformers library

```bash
pip install torch transformers datasets accelerate
```

## Approach 1: Pure Genesis Implementation

### 1. Model Configuration and Setup

```python
import genesis
import genesis.nn as nn
from genesis.models.qwen import ModelArgs, Transformer
import numpy as np

# Configure Qwen model for educational purposes
config = ModelArgs(
    block_size=2048,          # Context length
    vocab_size=32000,         # Vocabulary size
    n_layer=12,               # Number of transformer layers
    num_attention_heads=12,   # Number of attention heads
    hidden_size=768,          # Hidden dimension
    intermediate_size=3072,   # Feed-forward dimension
    num_key_value_heads=12,   # Key-value heads (for GQA)
    head_dim=64,              # Attention head dimension
    rope_base=10000,          # RoPE base frequency
    max_position_embeddings=2048
)

print(f"Model configuration:")
print(f"  Layers: {config.n_layer}")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Attention heads: {config.num_attention_heads}")
print(f"  Vocabulary size: {config.vocab_size}")
```

### 2. Model Instantiation

```python
# Create Qwen model using Genesis
model = Transformer(config)

# Calculate model parameters
total_params = sum(p.data.size for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.1f}M")

# Initialize weights
def init_weights(module):
    """Initialize model weights"""
    if isinstance(module, nn.Linear):
        # Initialize weights with small random values
        module.weight.data = genesis.randn(*module.weight.shape) * 0.02
        if module.bias is not None:
            module.bias.data = genesis.zeros(*module.bias.shape)

# Apply weight initialization
model.apply(init_weights)
print("Model weights initialized")
```

### 3. Simple Data Preparation

```python
# Simple synthetic data for demonstration
class SimpleTextDataset:
    """Simple dataset for language modeling"""
    
    def __init__(self, vocab_size=32000, seq_length=512, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        # Generate random token sequences
        self.data = genesis.tensor(
            np.random.randint(0, vocab_size, (num_samples, seq_length))
        )
        
    def __len__(self):
        return self.num_samples
    
    def get_batch(self, batch_size=4, start_idx=0):
        """Get a batch of sequences"""
        end_idx = min(start_idx + batch_size, self.num_samples)
        batch_data = self.data[start_idx:end_idx]
        
        # For language modeling: input = tokens[:-1], target = tokens[1:]
        input_ids = batch_data[:, :-1]
        labels = batch_data[:, 1:]
        
        return input_ids, labels

# Create dataset
dataset = SimpleTextDataset(vocab_size=config.vocab_size, seq_length=512, num_samples=100)
print(f"Dataset created with {len(dataset)} samples")
```

### 4. Training Setup

```python
import genesis.optim as optim

# Set up optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.SoftmaxLoss()

print(f"Optimizer: {type(optimizer).__name__}")
print(f"Learning rate: 1e-4")
print(f"Loss function: {type(criterion).__name__}")
```

### 5. Training Loop

```python
def train_step(model, input_ids, labels, criterion, optimizer):
    """Single training step"""
    # Forward pass
    logits = model(input_ids)
    
    # Reshape for loss calculation
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    labels_flat = labels.view(batch_size * seq_len)
    
    # Calculate loss
    loss = criterion(logits_flat, labels_flat)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.data ** 2
    total_norm = total_norm ** 0.5
    
    # Clip gradients
    clip_coef = min(1.0, 1.0 / max(total_norm, 1e-6))
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = p.grad.data * clip_coef
    
    # Update weights
    optimizer.step()
    
    return loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)

# Training configuration
num_epochs = 5
batch_size = 2  # Small batch size for demo
log_interval = 10

print("Starting Genesis Qwen training...")
print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
print("-" * 50)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = len(dataset) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        input_ids, labels = dataset.get_batch(batch_size, start_idx)
        
        # Training step
        loss = train_step(model, input_ids, labels, criterion, optimizer)
        total_loss += loss
        
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    print("-" * 30)

print("Pure Genesis training completed!")
```

## Approach 2: Production Hybrid Training

For production use, Genesis integrates with PyTorch and Transformers library. This is the approach used in `apps/llm/train_sft_qwen.py`.

### 1. Setup and Dependencies

```python
import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import os

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer loaded: {tokenizer.name_or_path}")
print(f"Vocabulary size: {len(tokenizer)}")
```

### 2. Data Preparation for SFT

```python
class ConversationDataset(Dataset):
    """Dataset for supervised fine-tuning with conversation format"""
    
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = self.load_conversations(data_path)
        
    def load_conversations(self, data_path):
        """Load conversation data from JSON lines"""
        conversations = []
        
        # Example conversation format
        sample_conversations = [
            {
                "messages": [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain neural networks"},
                    {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks..."}
                ]
            }
        ]
        
        return sample_conversations
    
    def format_conversation(self, messages):
        """Format conversation into training text"""
        text = ""
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"].strip()
            
            if i != len(messages) - 1:
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            else:
                # Last message (assistant response)
                text += f"<|im_start|>{role}\n{content}<|im_end|>"
        
        return text
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        text = self.format_conversation(conversation["messages"])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Create dataset
dataset = ConversationDataset("./data", tokenizer)
print(f"Dataset created with {len(dataset)} conversations")
```

### 3. Model Setup with Transformers

```python
# Load pre-trained Qwen model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Model loaded: {model.config.model_type}")
print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")
```

### 4. Training Configuration

```python
# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen_sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=1000,
    learning_rate=5e-5,
    fp16=True,  # Mixed precision training
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,  # Disable wandb logging
    gradient_checkpointing=True,
    dataloader_drop_last=True,
    remove_unused_columns=False,
)

print("Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Mixed precision: {training_args.fp16}")
```

### 5. Trainer Setup and Training

```python
# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Using same dataset for demo
    tokenizer=tokenizer,
)

print("Trainer created successfully")

# Start training
print("Starting supervised fine-tuning...")
trainer.train()

# Save the final model
trainer.save_model("./qwen_sft_final")
tokenizer.save_pretrained("./qwen_sft_final")

print("Training completed and model saved!")
```

### 6. Inference with Trained Model

```python
# Load trained model for inference
from transformers import pipeline

# Create text generation pipeline
generator = pipeline(
    "text-generation",
    model="./qwen_sft_final",
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Test the model
test_prompt = "<|im_start|>user\nWhat is artificial intelligence?<|im_end|>\n<|im_start|>assistant\n"

response = generator(
    test_prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

print("Generated response:")
print(response[0]["generated_text"])
```

## Advanced Features

### Mixed Precision Training

Genesis supports mixed precision training for both approaches:

```python
# For pure Genesis (in development)
genesis.enable_autocast = True

# For hybrid approach (using PyTorch)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Learning Rate Scheduling

```python
# Genesis optimizer with learning rate scheduling
from genesis.optim.lr_scheduler import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)

# Update learning rate after each step
scheduler.step()
```

### Model Checkpointing

```python
# Save Genesis model
genesis.save_checkpoint({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'checkpoint.pkl')

# Load Genesis model
checkpoint = genesis.load_checkpoint('checkpoint.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Performance Tips

1. **Batch Size**: Start with smaller batch sizes (2-4) for Genesis implementation
2. **Gradient Accumulation**: Use gradient accumulation for effective larger batch sizes
3. **Mixed Precision**: Enable FP16 to reduce memory usage and increase speed
4. **Gradient Clipping**: Prevent gradient explosion in transformer training
5. **Learning Rate**: Use warmup and cosine decay scheduling

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Gradient Explosion**: Enable gradient clipping
3. **Slow Convergence**: Check learning rate and warmup schedule
4. **NaN Loss**: Reduce learning rate or check data quality

### Memory Optimization

```python
# Reduce memory usage
- Use smaller batch sizes
- Enable gradient checkpointing
- Use mixed precision (FP16)
- Reduce sequence length
- Use gradient accumulation instead of large batches
```

## Next Steps

1. **Scale up**: Try larger models and datasets
2. **Fine-tuning**: Experiment with different fine-tuning strategies
3. **Evaluation**: Implement proper evaluation metrics
4. **Deployment**: Set up inference pipelines
5. **Optimization**: Profile and optimize training performance

This guide demonstrates both the educational pure Genesis approach and the production-ready hybrid approach used in real applications.