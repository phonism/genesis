# Large Language Model Training Guide

This comprehensive guide covers training large language models (LLMs) using Genesis, from basic setup to advanced optimization techniques. We'll use the Qwen model as our primary example.

## Overview

Genesis provides a complete framework for training transformer-based language models with features like:
- Qwen model architecture implementation
- Mixed precision training (FP16/BF16)  
- Gradient clipping and learning rate scheduling
- Distributed training support
- Efficient checkpoint management

## Prerequisites

- GPU with at least 16GB VRAM (A100/A800 recommended)
- CUDA 11.8+ and appropriate drivers
- Python 3.8+ with Genesis installed

## Quick Start

### 1. Basic Qwen Model Setup

```python
import genesis
import genesis.nn as nn
from genesis.models.qwen import QwenConfig, QwenModel

# Configure model
config = QwenConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_attention_heads=16,
    num_hidden_layers=24,
    intermediate_size=5632,
    max_position_embeddings=2048,
    dtype=genesis.float16  # Use mixed precision
)

# Create model
model = QwenModel(config)
print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")
```

### 2. Data Preparation

```python
import genesis
from torch.utils.data import DataLoader

class TextDataset:
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize and pad/truncate
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Convert to Genesis tensor
        input_ids = genesis.tensor(tokens[:-1], dtype=genesis.int64)
        labels = genesis.tensor(tokens[1:], dtype=genesis.int64)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

# Load your dataset
dataset = TextDataset(train_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 3. Training Setup

```python
import genesis.optim as optim
import genesis.nn as nn

# Move model to GPU
device = genesis.cuda()
model = model.to(device)

# Setup optimizer with weight decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8
)

# Learning rate scheduler with warmup
total_steps = len(dataloader) * num_epochs
warmup_steps = total_steps // 10

scheduler = optim.get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Loss function
criterion = nn.CrossEntropyLoss()
```

## Training Loop Implementation

### Basic Training Loop

```python
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        if batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()
            print(f'Batch {batch_idx}: loss={loss.item():.4f}, lr={current_lr:.2e}')
    
    return total_loss / num_batches

# Training loop
for epoch in range(num_epochs):
    avg_loss = train_epoch(model, dataloader, optimizer, scheduler, criterion, device)
    print(f'Epoch {epoch}: average loss = {avg_loss:.4f}')
    
    # Save checkpoint
    if epoch % 10 == 0:
        genesis.save_checkpoint(
            model.state_dict(),
            optimizer.state_dict(),
            f'qwen_checkpoint_epoch_{epoch}.pth'
        )
```

### Advanced Training with Mixed Precision

```python
import genesis

def train_epoch_mixed_precision(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    
    # Enable mixed precision
    genesis.enable_autocast = True
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with autocast
        with genesis.autocast():
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: loss={loss.item():.4f}')
    
    return total_loss / len(dataloader)
```

## Advanced Training Techniques

### 1. Gradient Accumulation

```python
def train_with_gradient_accumulation(model, dataloader, optimizer, scheduler, 
                                   criterion, device, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Compute loss and scale by accumulation steps
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        if batch_idx % (100 * accumulation_steps) == 0:
            print(f'Batch {batch_idx}: loss={loss.item() * accumulation_steps:.4f}')
    
    return total_loss / len(dataloader)
```

### 2. Dynamic Loss Scaling

```python
class DynamicLossScaler:
    def __init__(self, init_scale=2**16, scale_factor=2.0, scale_window=1000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._growth_tracker = 0
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale
    
    def step(self, optimizer, has_overflow=False):
        if has_overflow:
            self.scale = max(self.scale / self.scale_factor, 1.0)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.scale_window:
                self.scale *= self.scale_factor
                self._growth_tracker = 0
        
        return not has_overflow

# Usage in training loop
scaler = DynamicLossScaler()

for batch in dataloader:
    # Forward pass
    loss = compute_loss(model, batch)
    scaled_loss = scaler.scale_loss(loss)
    
    # Backward pass
    optimizer.zero_grad()
    scaled_loss.backward()
    
    # Check for overflow
    has_overflow = check_gradient_overflow(model.parameters())
    
    # Unscale and step
    scaler.unscale_gradients(optimizer)
    should_step = scaler.step(optimizer, has_overflow)
    
    if should_step:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
```

### 3. Checkpointing and Resume Training

```python
class TrainingManager:
    def __init__(self, model, optimizer, scheduler, save_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.current_epoch = 0
        self.best_loss = float('inf')
    
    def save_checkpoint(self, epoch, loss, metrics=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'metrics': metrics or {}
        }
        
        # Save regular checkpoint
        checkpoint_path = f'{self.save_dir}/checkpoint_epoch_{epoch}.pth'
        genesis.save(checkpoint, checkpoint_path)
        
        # Save best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = f'{self.save_dir}/best_model.pth'
            genesis.save(checkpoint, best_path)
            print(f"New best model saved with loss: {loss:.4f}")
        
        # Save latest
        latest_path = f'{self.save_dir}/latest_checkpoint.pth'
        genesis.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = genesis.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Resumed from epoch {self.current_epoch}, best loss: {self.best_loss:.4f}")
        return checkpoint

# Usage
training_manager = TrainingManager(model, optimizer, scheduler)

# Resume from checkpoint if exists
try:
    training_manager.load_checkpoint('checkpoints/latest_checkpoint.pth')
except FileNotFoundError:
    print("Starting training from scratch")

# Training loop with checkpointing
for epoch in range(training_manager.current_epoch, num_epochs):
    avg_loss = train_epoch(model, dataloader, optimizer, scheduler, criterion, device)
    
    # Save checkpoint
    training_manager.save_checkpoint(epoch, avg_loss)
    
    print(f'Epoch {epoch}: average loss = {avg_loss:.4f}')
```

## Model Evaluation and Inference

### 1. Model Evaluation

```python
def evaluate_model(model, eval_dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with genesis.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item() * shift_labels.numel()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = genesis.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity.item()
    }

# Evaluate model
eval_metrics = evaluate_model(model, eval_dataloader, criterion, device)
print(f"Eval Loss: {eval_metrics['loss']:.4f}, Perplexity: {eval_metrics['perplexity']:.2f}")
```

### 2. Text Generation

```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = genesis.tensor([input_ids], dtype=genesis.int64).to(device)
    
    generated_ids = input_ids.copy()
    
    with genesis.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-p filtering
            sorted_logits, sorted_indices = genesis.sort(logits, descending=True)
            cumulative_probs = genesis.cumsum(genesis.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = genesis.softmax(logits, dim=-1)
            next_token = genesis.multinomial(probs, 1).item()
            
            # Add to generated sequence
            generated_ids.append(next_token)
            
            # Update input tensor
            input_tensor = genesis.tensor([generated_ids], dtype=genesis.int64).to(device)
            
            # Check for end token
            if next_token == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

# Generate text
prompt = "The future of artificial intelligence is"
generated = generate_text(model, tokenizer, prompt, max_length=50)
print(f"Generated: {generated}")
```

## Production Deployment

### 1. Model Optimization for Inference

```python
def optimize_for_inference(model, save_path):
    """Optimize model for production inference."""
    model.eval()
    
    # Create inference-optimized state
    inference_state = {
        'model_state_dict': model.state_dict(),
        'model_config': model.config.__dict__,
        'inference_optimized': True,
        'genesis_version': genesis.__version__
    }
    
    genesis.save(inference_state, save_path)
    print(f"Inference-optimized model saved to {save_path}")

def load_for_inference(model_path, device=None):
    """Load model optimized for inference."""
    checkpoint = genesis.load(model_path)
    config = QwenConfig(**checkpoint['model_config'])
    
    # Create model
    model = QwenModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if device:
        model = model.to(device)
    
    return model

# Optimize and save
optimize_for_inference(model, 'qwen_inference.pth')

# Load for inference
inference_model = load_for_inference('qwen_inference.pth', device=genesis.cuda())
```

### 2. Inference Server Setup

```python
class LLMInferenceServer:
    def __init__(self, model_path, tokenizer, device=None):
        self.tokenizer = tokenizer
        self.device = device or genesis.cuda()
        self.model = load_for_inference(model_path, self.device)
    
    def generate(self, prompt, max_length=100, temperature=0.8, top_p=0.9):
        """Generate text from prompt."""
        return generate_text(
            self.model, self.tokenizer, prompt,
            max_length=max_length, temperature=temperature, top_p=top_p
        )
    
    def batch_generate(self, prompts, max_length=100, temperature=0.8, top_p=0.9):
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_length, temperature, top_p)
            results.append(result)
        return results

# Create inference server
server = LLMInferenceServer('qwen_inference.pth', tokenizer)

# Generate responses
responses = server.batch_generate([
    "What is the meaning of life?",
    "Explain quantum computing in simple terms.",
    "Write a short story about AI."
])
```

## Performance Optimization Tips

### 1. Memory Optimization
- Use gradient checkpointing for large models
- Enable mixed precision training (FP16/BF16)
- Use gradient accumulation for large effective batch sizes
- Regularly clear GPU cache

### 2. Training Speed
- Use appropriate batch sizes for your GPU
- Enable compiled mode if available
- Use efficient data loading with multiple workers
- Profile training to identify bottlenecks

### 3. Model Quality
- Use proper learning rate scheduling
- Apply gradient clipping to stabilize training
- Monitor training metrics closely
- Use validation sets to prevent overfitting

This guide provides a comprehensive foundation for training LLMs with Genesis. Adapt the techniques based on your specific model size, dataset, and computational resources.