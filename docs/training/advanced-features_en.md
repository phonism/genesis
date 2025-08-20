# Advanced Training Features

Genesis provides several advanced features to enhance training efficiency and model performance.

## 🚀 Mixed Precision Training (AMP)

Automatic Mixed Precision (AMP) allows you to train models faster with lower memory usage by utilizing FP16/BF16 computations where appropriate while maintaining FP32 master weights for numerical stability.

### Basic Usage

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
from genesis.amp import autocast, GradScaler

# Create model and optimizer
model = nn.Linear(1024, 512)
optimizer = optim.Adam(model.parameters())

# Initialize gradient scaler for mixed precision
scaler = GradScaler()

# Training loop with AMP
for data, target in dataloader:
    optimizer.zero_grad()
    
    # Use autocast for automatic mixed precision
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Scale the loss and backward pass
    scaler.scale(loss).backward()
    
    # Unscale and step optimizer
    scaler.step(optimizer)
    scaler.update()
```

### Supported Data Types

Genesis supports multiple precision formats:

- **float16 (FP16)**: Half precision, fastest on most GPUs
- **bfloat16 (BF16)**: Brain floating point, better range than FP16
- **float32 (FP32)**: Single precision, default for master weights

### Benefits

- **Speed**: Up to 2x faster training on modern GPUs
- **Memory**: Reduced memory usage allows larger batch sizes
- **Accuracy**: Maintains model accuracy with loss scaling

## ✂️ Gradient Clipping

Gradient clipping helps prevent gradient explosion in deep networks and improves training stability, especially for RNNs and Transformers.

### Gradient Norm Clipping

Clips gradients when their L2 norm exceeds a threshold:

```python
import genesis.nn.utils as nn_utils

# During training
loss.backward()

# Clip gradients by norm (recommended for most cases)
nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

### Gradient Value Clipping

Clips gradient values to a specific range:

```python
# Clip gradients by value
nn_utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### When to Use

- **Gradient Norm Clipping**: Recommended for RNNs, LSTMs, and Transformers
- **Gradient Value Clipping**: Useful when you need hard limits on gradient values
- **Typical Values**: max_norm between 0.5 and 5.0 for most models

## 📈 Learning Rate Schedulers

Learning rate schedulers adjust the learning rate during training to improve convergence and final model performance.

### StepLR

Decays learning rate by gamma every step_size epochs:

```python
from genesis.optim.lr_scheduler import StepLR

optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # Decay LR every 30 epochs
```

### ExponentialLR

Decays learning rate exponentially:

```python
from genesis.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(optimizer, gamma=0.95)

for epoch in range(100):
    train(...)
    scheduler.step()  # LR = LR * 0.95 each epoch
```

### CosineAnnealingLR

Uses cosine annealing schedule:

```python
from genesis.optim.lr_scheduler import CosineAnnealingLR

# T_max: Maximum number of iterations
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

for epoch in range(100):
    train(...)
    scheduler.step()
```

### Custom Learning Rate Schedule

You can also implement custom schedules:

```python
def custom_lr_lambda(epoch):
    # Warmup for first 10 epochs, then decay
    if epoch < 10:
        return epoch / 10
    else:
        return 0.95 ** (epoch - 10)

scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_lambda)
```

## 💾 Checkpointing

Save and restore model states during training for fault tolerance and model deployment.

### Saving Checkpoints

```python
import genesis

# Save model state
genesis.save_checkpoint({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_accuracy': best_acc
}, 'checkpoint_epoch_10.pth')
```

### Loading Checkpoints

```python
# Load checkpoint
checkpoint = genesis.load_checkpoint('checkpoint_epoch_10.pth')

# Restore model and optimizer states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### Best Practices

1. **Regular Saving**: Save checkpoints every N epochs
2. **Best Model Tracking**: Keep the best performing model
3. **Metadata Storage**: Include training configuration and metrics

```python
# Example: Save best model during training
best_loss = float('inf')

for epoch in range(num_epochs):
    val_loss = validate(model, val_loader)
    
    if val_loss < best_loss:
        best_loss = val_loss
        genesis.save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss
        }, 'best_model.pth')
```

## 🔧 Complete Training Example

Here's a complete example combining all advanced features:

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim
from genesis.amp import autocast, GradScaler
from genesis.optim.lr_scheduler import CosineAnnealingLR
import genesis.nn.utils as nn_utils

# Model setup
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
scaler = GradScaler()

# Training configuration
max_grad_norm = 1.0
checkpoint_interval = 10

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()
    
    # Update learning rate
    scheduler.step()
    
    # Save checkpoint
    if epoch % checkpoint_interval == 0:
        genesis.save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, f'checkpoint_epoch_{epoch}.pth')
```

## 📊 Performance Tips

### Memory Optimization
- Use gradient accumulation for larger effective batch sizes
- Enable gradient checkpointing for very deep models
- Use mixed precision training to reduce memory usage

### Speed Optimization
- Use the appropriate data type (FP16 for speed, BF16 for stability)
- Tune gradient accumulation steps
- Profile your training loop to identify bottlenecks

### Convergence Tips
- Start with a learning rate finder to identify optimal LR
- Use warmup for large batch training
- Monitor gradient norms to detect instability early

## 🔗 Related Topics

- [Basic Training Tutorial](../tutorials/basic-training.md)
- [Performance Tuning Guide](../tutorials/performance-tuning.md)
- [Model Architecture Guide](../core-components/index.md)
- [Optimizer Documentation](../api/optim/optimizers.md)