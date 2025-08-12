# Optimizers (genesis.optim)

## Overview

The `genesis.optim` module provides optimizers for training neural networks. It implements state-of-the-art optimization algorithms with support for parameter groups, gradient clipping, and mixed precision training.

## Core Concepts

### Optimization Process

Optimizers update model parameters based on computed gradients using various algorithms:
1. **Gradient Descent**: Basic parameter update using gradients
2. **Momentum**: Accelerated convergence using moving averages
3. **Adaptive Learning Rates**: Different learning rates per parameter
4. **Regularization**: Weight decay and gradient clipping

### Parameter Groups

Parameters can be organized into groups with different hyperparameters:
- Different learning rates for different layers
- Selective weight decay application
- Layer-specific optimization settings

## Base Classes

### `optim.Optimizer`

Abstract base class for all optimizers.

```python
class Optimizer:
    """
    Base class for all optimizers.
    
    Args:
        params: Iterable of parameters or dicts defining parameter groups
        defaults: Dict containing default values for optimization options
    """
    
    def __init__(self, params, defaults: dict):
        """
        Initialize the optimizer.
        
        Args:
            params: Model parameters or parameter groups
            defaults: Default hyperparameter values
        """
```

#### Core Methods

##### Optimization Step
```python
def step(self, closure: Optional[Callable] = None) -> Optional[float]:
    """
    Perform a single optimization step.
    
    Args:
        closure: Optional function to reevaluate the model and return loss
        
    Returns:
        Loss value if closure is provided
        
    Example:
        >>> optimizer.zero_grad()
        >>> loss = criterion(output, target)
        >>> loss.backward()
        >>> optimizer.step()
    """

def zero_grad(self, set_to_none: bool = True) -> None:
    """
    Clear gradients of all optimized parameters.
    
    Args:
        set_to_none: If True, set gradients to None instead of zero
        
    Example:
        >>> # Clear gradients before each training step
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
```

##### State Management
```python
def state_dict(self) -> Dict[str, Any]:
    """
    Return optimizer state as a dictionary.
    
    Returns:
        Dictionary containing optimizer state and parameter groups
        
    Example:
        >>> # Save optimizer state
        >>> state = optimizer.state_dict()
        >>> genesis.save(state, 'optimizer_checkpoint.pth')
    """

def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
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

##### Parameter Groups
```python
def add_param_group(self, param_group: Dict[str, Any]) -> None:
    """
    Add a parameter group to the optimizer.
    
    Args:
        param_group: Dictionary specifying parameters and their options
        
    Example:
        >>> # Add new layer with different learning rate
        >>> optimizer.add_param_group({
        ...     'params': new_layer.parameters(),
        ...     'lr': 0.001
        ... })
    """

@property
def param_groups(self) -> List[Dict[str, Any]]:
    """
    Access parameter groups.
    
    Returns:
        List of parameter group dictionaries
        
    Example:
        >>> # Manually adjust learning rates
        >>> for group in optimizer.param_groups:
        ...     group['lr'] *= 0.9
    """
```

## Optimizers

### `optim.SGD`

Stochastic Gradient Descent optimizer with momentum and weight decay.

```python
class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (required)
        momentum: Momentum factor (default: 0)
        dampening: Dampening for momentum (default: 0)
        weight_decay: Weight decay coefficient (default: 0)
        nesterov: Whether to use Nesterov momentum (default: False)
        
    Algorithm:
        v_t = momentum * v_{t-1} + g_t
        p_t = p_{t-1} - lr * v_t
        
    Where:
        g_t: gradient at time t
        v_t: velocity at time t
        p_t: parameters at time t
    """
    
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False
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

# Nesterov accelerated gradient
optimizer = optim.SGD(model.parameters(), lr=0.01,
                     momentum=0.9, nesterov=True)

# Different learning rates for different layers
optimizer = optim.SGD([
    {'params': model.features.parameters(), 'lr': 0.001},
    {'params': model.classifier.parameters(), 'lr': 0.01}
], momentum=0.9)
```

### `optim.Adam`

Adaptive Moment Estimation optimizer combining RMSprop and momentum.

```python
class Adam(Optimizer):
    """
    Adam optimizer.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient
               and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
        amsgrad: Whether to use AMSGrad variant (default: False)
        
    Algorithm:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁ᵗ)
        v̂_t = v_t / (1 - β₂ᵗ)
        p_t = p_{t-1} - lr * m̂_t / (√v̂_t + ε)
        
    Where:
        g_t: gradient
        m_t: first moment estimate (momentum)
        v_t: second moment estimate (adaptive learning rate)
        m̂_t, v̂_t: bias-corrected moment estimates
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
```

#### State Variables

Each parameter maintains the following state:
- `step`: Number of optimization steps taken
- `exp_avg`: Exponential moving average of gradient values (momentum)
- `exp_avg_sq`: Exponential moving average of squared gradient values
- `max_exp_avg_sq`: Maximum of exp_avg_sq (AMSGrad only)

#### Usage Examples

```python
# Default Adam (most common)
optimizer = optim.Adam(model.parameters())

# Custom learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Transformer model settings
optimizer = optim.Adam(model.parameters(), lr=0.0001,
                      betas=(0.9, 0.98), eps=1e-9)

# With weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001,
                      weight_decay=1e-5)

# Fine-tuning with different learning rates
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.decoder.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3}
])

# Using AMSGrad variant
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
```

### `optim.AdamW`

Adam optimizer with decoupled weight decay.

```python
class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        amsgrad: Whether to use AMSGrad variant (default: False)
        
    Difference from Adam:
        Adam: p_t = p_{t-1} - lr * (m̂_t / (√v̂_t + ε) + wd * p_{t-1})
        AdamW: p_t = p_{t-1} * (1 - lr * wd) - lr * m̂_t / (√v̂_t + ε)
        
    AdamW decouples weight decay from gradient computation, applying it
    directly to parameters for better regularization.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
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
                       betas=(0.9, 0.95), weight_decay=0.1)

# Exclude bias and normalization from weight decay
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if any(nd in name for nd in ['bias', 'norm', 'ln']):
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=1e-4)
```

### `optim.RMSprop`

Root Mean Square Propagation optimizer.

```python
class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-2)
        alpha: Smoothing constant (default: 0.99)
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
        momentum: Momentum factor (default: 0)
        centered: Whether to normalize by centered second moment (default: False)
    """
```

## Gradient Clipping

Utilities to prevent gradient explosion.

```python
def clip_grad_norm_(
    parameters: Iterable[Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> float:
    """
    Clip gradients by global norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum gradient norm
        norm_type: Type of norm (1, 2, or inf)
        error_if_nonfinite: Error if total norm is non-finite
        
    Returns:
        Total norm of gradients before clipping
        
    Example:
        >>> loss.backward()
        >>> # Clip gradients to prevent explosion
        >>> genesis.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        >>> optimizer.step()
    """

def clip_grad_value_(
    parameters: Iterable[Tensor],
    clip_value: float
) -> None:
    """
    Clip gradients by value.
    
    Args:
        parameters: Iterable of parameters with gradients
        clip_value: Clipping threshold
        
    Example:
        >>> loss.backward()
        >>> # Limit gradients to [-1, 1] range
        >>> genesis.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        >>> optimizer.step()
    """
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
criterion = nn.CrossEntropyLoss()

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
        
        # Gradient clipping (optional)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```

### Advanced Training with Mixed Precision

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
        
        # Use autocast for mixed precision
        with genesis.autocast():
            outputs = model(batch['input'])
            loss = criterion(outputs, batch['target'])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
```

### Learning Rate Scheduling

```python
from genesis.optim.lr_scheduler import CosineAnnealingLR

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

for epoch in range(num_epochs):
    # Training
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    scheduler.step()
    print(f'Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}')
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
- **Disadvantages**: Higher memory usage, potential overfitting
- **Best for**:
  - NLP tasks
  - Rapid prototyping
  - Sparse gradients
- **Recommended settings**: `lr=1e-3, betas=(0.9, 0.999)`

### AdamW
- **Advantages**: Better generalization than Adam, excellent for large models
- **Disadvantages**: Higher memory usage
- **Best for**:
  - Transformer models (BERT, GPT)
  - Large-scale pre-training
  - When strong regularization is needed
- **Recommended settings**: `lr=5e-5, weight_decay=0.01`

### RMSprop
- **Advantages**: Good for non-stationary objectives
- **Disadvantages**: Can be unstable with high learning rates
- **Best for**:
  - RNN training
  - Reinforcement learning
  - Non-stationary problems

## Performance Tips

1. **Gradient Accumulation**: Simulate larger batch sizes when memory is limited
2. **Gradient Clipping**: Essential for RNNs and Transformers
3. **Parameter Groups**: Use different learning rates for different layers
4. **Weight Decay**: AdamW generally performs better than Adam + L2 regularization
5. **Learning Rate Warmup**: Use warmup for large batch training
6. **Mixed Precision**: Reduces memory usage and speeds up training

## Memory Considerations

- Optimizers maintain state per parameter (Adam/AdamW use 2x parameter memory)
- Use `zero_grad(set_to_none=True)` to reduce memory fragmentation
- Consider optimizer state when moving models between devices
- Save optimizer state in checkpoints for resuming training

## Best Practices

1. **Always clear gradients** before backward pass
2. **Use gradient clipping** for RNNs and Transformers
3. **Monitor learning rates** throughout training
4. **Save optimizer state** in checkpoints
5. **Use appropriate weight decay** for your model type
6. **Consider mixed precision** for large models

## See Also

- [Learning Rate Schedulers](schedulers.md) - Dynamic learning rate adjustment
- [Neural Network Modules](../nn/modules.md) - Building models
- [Autograd](../autograd.md) - Automatic differentiation
- [Examples](../../../samples/) - Complete training examples