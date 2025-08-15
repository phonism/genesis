# Learning Rate Schedulers

Genesis provides learning rate schedulers to adjust the learning rate during training, which is crucial for achieving optimal convergence in deep learning models.

## Overview

Learning rate scheduling is a technique used to adjust the learning rate throughout the training process. Genesis provides PyTorch-compatible learning rate schedulers that can significantly improve model convergence.

## Available Schedulers

### LambdaLR

The `LambdaLR` scheduler allows you to define a custom function to modify the learning rate at each epoch.

```python
import genesis.optim as optim

class LambdaLR:
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        """
        Multiply learning rate by a factor given by lr_lambda function.
        
        Args:
            optimizer: Wrapped optimizer
            lr_lambda: Function or list of functions to compute multiplicative factor
            last_epoch: The index of last epoch
            verbose: If True, prints a message for each update
        """
```

**Usage Example:**
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Create model and optimizer
model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define learning rate schedule function
def lr_lambda(epoch):
    # Decay learning rate by factor of 0.95 every 10 epochs
    return 0.95 ** (epoch // 10)

# Create scheduler
scheduler = optim.LambdaLR(optimizer, lr_lambda=lr_lambda)

# Training loop
for epoch in range(100):
    # Training code here
    loss = train_one_epoch(model, dataloader, optimizer)
    
    # Step scheduler
    scheduler.step()
    print(f"Epoch {epoch}: lr={scheduler.get_last_lr()}")
```

### Cosine Annealing with Warmup

The `get_cosine_schedule_with_warmup` function creates a cosine annealing schedule with linear warmup.

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a schedule with linear warmup and cosine decay.
    
    Args:
        optimizer: Wrapped optimizer
        num_warmup_steps: Number of steps for warmup phase
        num_training_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler object
    """
```

**Usage Example:**
```python
import genesis.optim as optim

# Training configuration
num_epochs = 100
steps_per_epoch = 1000
total_steps = num_epochs * steps_per_epoch
warmup_steps = total_steps // 10  # 10% warmup

# Create optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
scheduler = optim.get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Training loop with per-step scheduling
step = 0
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass and optimization
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step scheduler every batch
        scheduler.step()
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}: lr={scheduler.get_last_lr():.6f}")
```

## Scheduler Methods

All schedulers provide the following methods:

### step()
```python
def step(self, epoch=None):
    """
    Update learning rate according to schedule.
    
    Args:
        epoch: Current epoch (optional, uses internal counter if None)
    """
```

### get_last_lr()
```python
def get_last_lr(self):
    """
    Return the last computed learning rate.
    
    Returns:
        Current learning rate value
    """
```

### state_dict()
```python
def state_dict(self):
    """
    Return the state of the scheduler as a dict.
    
    Returns:
        Dictionary containing scheduler state
    """
```

### load_state_dict()
```python
def load_state_dict(self, state_dict):
    """
    Load scheduler state from dict.
    
    Args:
        state_dict: Dictionary containing scheduler state
    """
```

## Common Patterns

### Exponential Decay
```python
# Decay learning rate by 0.95 every epoch
scheduler = optim.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
```

### Step Decay
```python
# Reduce learning rate by half every 30 epochs
def step_decay(epoch):
    return 0.5 ** (epoch // 30)

scheduler = optim.LambdaLR(optimizer, lr_lambda=step_decay)
```

### Polynomial Decay
```python
# Polynomial decay to zero
def poly_decay(epoch, total_epochs=100, power=0.9):
    return (1 - epoch / total_epochs) ** power

scheduler = optim.LambdaLR(optimizer, lr_lambda=lambda epoch: poly_decay(epoch))
```

### Cosine with Restarts
```python
import math

def cosine_restart(epoch, restart_period=50):
    epoch_in_cycle = epoch % restart_period
    return 0.5 * (1 + math.cos(math.pi * epoch_in_cycle / restart_period))

scheduler = optim.LambdaLR(optimizer, lr_lambda=cosine_restart)
```

## Integration with Training

### Basic Training Loop
```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Setup
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=10000
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        # Logging
        if batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()
            print(f'Epoch: {epoch}, Batch: {batch_idx}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}')
```

### Checkpoint Integration
```python
import genesis

# Save scheduler state with model checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'loss': loss
}
genesis.save_checkpoint(checkpoint, 'checkpoint.pth')

# Load scheduler state
checkpoint = genesis.load_checkpoint('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

## Best Practices

1. **Choose the Right Schedule**: 
   - Use cosine annealing for most applications
   - Add warmup for transformer models
   - Use step decay for fine-tuning

2. **Warmup Phase**:
   - Essential for large batch sizes
   - Recommended for transformer architectures
   - Typically 5-10% of total training steps

3. **Monitoring**:
   - Log learning rate values
   - Plot learning rate schedule
   - Monitor validation loss during training

4. **Checkpointing**:
   - Always save scheduler state
   - Resume training with correct learning rate
   - Essential for long training runs

## Examples

### Transformer Training Schedule
```python
# Typical transformer training schedule
def get_transformer_schedule(optimizer, d_model=512, warmup_steps=4000):
    def lr_lambda(step):
        if step == 0:
            return 0
        return min(step ** -0.5, step * warmup_steps ** -1.5) * (d_model ** -0.5)
    
    return optim.LambdaLR(optimizer, lr_lambda=lr_lambda)

scheduler = get_transformer_schedule(optimizer, d_model=512, warmup_steps=4000)
```

### Learning Rate Range Test
```python
# Find optimal learning rate range
def lr_range_test(model, optimizer, start_lr=1e-7, end_lr=10, num_it=100):
    lrs = []
    losses = []
    
    lr_lambda = lambda step: (end_lr / start_lr) ** (step / num_it)
    scheduler = optim.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    for i in range(num_it):
        # Training step
        loss = train_step(model, batch)
        losses.append(loss)
        lrs.append(scheduler.get_last_lr())
        
        scheduler.step()
        
        if loss > 4 * min(losses):  # Stop if loss explodes
            break
    
    return lrs, losses
```

## Migration from PyTorch

Genesis learning rate schedulers are designed to be drop-in replacements for PyTorch schedulers:

```python
# PyTorch code
import torch.optim as optim
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Genesis equivalent
import genesis.optim as optim
scheduler = optim.LambdaLR(
    optimizer, 
    lr_lambda=lambda epoch: 0.5 * (1 + math.cos(math.pi * epoch / 100))
)
```

The API is compatible, making it easy to migrate existing PyTorch training scripts to Genesis.