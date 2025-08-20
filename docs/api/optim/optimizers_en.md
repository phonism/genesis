# Optimizers (genesis.optim)

## Overview

The `genesis.optim` module provides optimizers for training neural networks. It implements state-of-the-art optimization algorithms with support for momentum, weight decay, and adaptive learning rates.

## Core Concepts

### Optimization Process

Optimizers update model parameters based on computed gradients using various algorithms:
1. **Gradient Descent**: Basic parameter update using gradients
2. **Momentum**: Accelerated convergence using moving averages
3. **Adaptive Learning Rates**: Different learning rates per parameter
4. **Weight Decay**: L2 regularization

## Base Classes

### `optim.Optimizer`

Abstract base class for all optimizers.

```python
class Optimizer:
    """
    Base class for all optimizers.
    
    Provides common functionality for parameter updates, gradient zeroing,
    and state management across different optimization algorithms.
    """
    
    def __init__(self, params):
        """
        Initialize optimizer with parameters to optimize.
        
        Args:
            params: Iterable of parameters to optimize
        """
```

#### Core Methods

##### Optimization Step
```python
def step(self):
    """
    Perform a single optimization step (parameter update).
    
    Must be implemented by subclasses.
    
    Example:
        >>> optimizer.zero_grad()
        >>> loss = criterion(output, target)
        >>> loss.backward()
        >>> optimizer.step()
    """

def zero_grad(self):
    """
    Zero gradients of all optimized parameters.
    Sets grad to None for all parameters.
        
    Example:
        >>> # Clear gradients before each training step
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

def reset_grad(self):
    """
    Reset gradients of all optimized parameters.
    Alias for zero_grad().
    """
```

##### State Management
```python
def state_dict(self):
    """
    Return optimizer state as a dictionary.
    
    Returns:
        Dictionary containing optimizer state (excluding params)
        
    Example:
        >>> # Save optimizer state
        >>> state = optimizer.state_dict()
        >>> genesis.save(state, 'optimizer_checkpoint.pth')
    """

def load_state_dict(self, state_dict):
    """
    Load optimizer state from dictionary.
    
    Args:
        state_dict: Optimizer state dictionary
        
    Example:
        >>> # Restore optimizer state
        >>> state = genesis.load('optimizer_checkpoint.pth')
        >>> optimizer.load_state_dict(state)
    """
```

## Optimizers

### `optim.SGD`

Stochastic Gradient Descent optimizer with momentum and weight decay.

```python
class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum and weight decay.
    
    Implements the classical SGD algorithm with optional momentum for improved
    convergence and weight decay for regularization.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor, 0 disables momentum (default: 0.0)
        weight_decay: Weight decay (L2 penalty) factor (default: 0.0)
        
    Algorithm:
        grad = gradient + weight_decay * param
        velocity = momentum * velocity + (1 - momentum) * grad
        param = param - lr * velocity
    """
    
    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0
    ):
```

#### Usage Examples

```python
import genesis.optim as optim

# Basic SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum (recommended for most tasks)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD with weight decay
optimizer = optim.SGD(model.parameters(), lr=0.01, 
                     momentum=0.9, weight_decay=1e-4)
```

### `optim.Adam`

Adaptive Moment Estimation optimizer combining RMSprop and momentum.

```python
class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates.
    
    Implements the Adam algorithm which computes adaptive learning rates
    for each parameter using estimates of first and second moments of gradients.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.01)
        beta1: Coefficient for first moment estimate (default: 0.9)
        beta2: Coefficient for second moment estimate (default: 0.999)
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.0)
        
    Algorithm:
        grad = gradient + weight_decay * param
        m_t = β₁ * m_{t-1} + (1 - β₁) * grad
        v_t = β₂ * v_{t-1} + (1 - β₂) * grad²
        m̂_t = m_t / (1 - β₁ᵗ)
        v̂_t = v_t / (1 - β₂ᵗ)
        param = param - lr * m̂_t / (√v̂_t + ε)
    """
    
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0
    ):
```

#### State Variables

Each parameter maintains the following state:
- `t`: Time step counter (incremented on each step)
- `m`: First moment estimate (momentum)
- `v`: Second moment estimate (adaptive learning rate)

#### Usage Examples

```python
# Default Adam
optimizer = optim.Adam(model.parameters())

# Custom learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# With weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001,
                      weight_decay=1e-5)

# Transformer model settings
optimizer = optim.Adam(model.parameters(), lr=0.0001,
                      beta1=0.9, beta2=0.98, eps=1e-9)
```

### `optim.AdamW`

Adam optimizer with decoupled weight decay.

```python
class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.001)
        beta1: Coefficient for computing running averages (default: 0.9)
        beta2: Coefficient for computing running averages (default: 0.999)
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        
    Difference from Adam:
        Adam:  param = param - lr * (m̂_t / (√v̂_t + ε) + wd * param)
        AdamW: param = param - lr * (m̂_t / (√v̂_t + ε) + wd * param)
        
    AdamW applies weight decay directly to parameters rather than to gradients,
    providing better regularization especially for adaptive learning rate methods.
    """
    
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01
    ):
```

#### Usage Examples

```python
# Default AdamW (recommended for Transformers)
optimizer = optim.AdamW(model.parameters())

# BERT/GPT standard settings
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Large model training
optimizer = optim.AdamW(model.parameters(), lr=1e-4,
                       beta1=0.9, beta2=0.95, weight_decay=0.1)
```

## Training Examples

### Basic Training Loop

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Model and optimizer setup
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.SoftmaxLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```

### Training with Mixed Precision

```python
import genesis
import genesis.nn as nn
import genesis.optim as optim

# Enable mixed precision
genesis.enable_autocast = True

# Model setup
model = TransformerModel()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Mixed precision is handled automatically in Function.apply()
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
```

### Learning Rate Scheduling

```python
from genesis.optim.lr_scheduler import LambdaLR

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Define learning rate schedule
def lr_lambda(epoch):
    return 0.95 ** epoch

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(num_epochs):
    # Training
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    scheduler.step()
    print(f'Epoch {epoch}, LR: {scheduler.get_last_lr():.6f}')
```

### Gradient Accumulation

```python
# Simulate larger batch size through accumulation
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    # Forward pass
    outputs = model(batch['input'])
    loss = criterion(outputs, batch['target'])
    
    # Normalize loss by accumulation steps
    loss = loss / accumulation_steps
    loss.backward()
    
    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Optimizer Selection Guide

### SGD
- **Advantages**: Simple, memory efficient, good generalization
- **Disadvantages**: Slow convergence, sensitive to learning rate
- **Best for**:
  - Computer vision tasks (ResNet, VGG)
  - Memory-constrained environments
  - When best generalization is needed
- **Recommended settings**: `lr=0.1, momentum=0.9, weight_decay=1e-4`

### Adam
- **Advantages**: Fast convergence, adaptive, less sensitive to hyperparameters
- **Disadvantages**: Higher memory usage (2x parameters)
- **Best for**:
  - NLP tasks
  - Rapid prototyping
  - Sparse gradients
- **Recommended settings**: `lr=1e-3, beta1=0.9, beta2=0.999`

### AdamW
- **Advantages**: Better generalization than Adam, excellent for large models
- **Disadvantages**: Higher memory usage (2x parameters)
- **Best for**:
  - Transformer models (BERT, GPT)
  - Large-scale pre-training
  - When strong regularization is needed
- **Recommended settings**: `lr=5e-5, weight_decay=0.01`

## Performance Tips

1. **Gradient Accumulation**: Simulate larger batch sizes when memory is limited
2. **Learning Rate Scheduling**: Use schedulers to improve convergence
3. **Weight Decay**: AdamW generally performs better than Adam + L2 regularization
4. **Mixed Precision**: Reduces memory usage and speeds up training
5. **Zero Gradients**: Always clear gradients before backward pass

## Memory Considerations

- Adam/AdamW maintain state per parameter (2x parameter memory for momentum and variance)
- Use `zero_grad()` to clear gradients and free memory
- Consider optimizer state when moving models between devices
- Save optimizer state in checkpoints for resuming training

## Best Practices

1. **Always clear gradients** before backward pass
2. **Monitor learning rates** throughout training
3. **Save optimizer state** in checkpoints
4. **Use appropriate weight decay** for your model type
5. **Consider mixed precision** for large models

## See Also

- [Learning Rate Schedulers](lr_scheduler_en.md) - Dynamic learning rate adjustment
- [Neural Network Modules](../nn/modules_en.md) - Building models
- [Autograd](../autograd_en.md) - Automatic differentiation
- [Examples](../../../samples/) - Complete training examples