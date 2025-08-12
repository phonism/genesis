# Neural Network Modules (genesis.nn)

## Overview

The `genesis.nn` module provides all the building blocks needed to create deep learning models. It follows a modular design where complex models are built by composing simpler components.

## Core Concepts

### Module System

All neural network components inherit from `nn.Module`, which provides:
- Parameter management
- Device and dtype handling  
- State serialization
- Forward pass definition

### Parameters

Parameters are tensors that are automatically tracked and updated during training:
- Automatically registered when assigned as module attributes
- Included in `module.parameters()` for optimizer
- Saved/loaded with model state

## Base Classes

### `nn.Module`

The base class for all neural network modules.

```python
class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        """Initialize the module."""
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.training = True
```

#### Core Methods

##### Forward Pass
```python
def forward(self, *args, **kwargs) -> Tensor:
    """
    Define the forward pass computation.
    Must be overridden by subclasses.
    
    Example:
        >>> class MyModule(nn.Module):
        ...     def forward(self, x):
        ...         return x * 2
    """
    raise NotImplementedError

def __call__(self, *args, **kwargs) -> Tensor:
    """
    Make module callable. Calls forward() internally.
    
    Note: Always use module(input) instead of module.forward(input)
    """
```

##### Parameter Management
```python
def parameters(self) -> List[Tensor]:
    """
    Return all parameters in the module.
    
    Returns:
        List of parameter tensors
        
    Example:
        >>> model = nn.Linear(10, 5)
        >>> params = model.parameters()
        >>> print(len(params))  # 2 (weight and bias)
    """

def named_parameters(self) -> List[Tuple[str, Tensor]]:
    """
    Return parameters with their names.
    
    Returns:
        List of (name, parameter) tuples
        
    Example:
        >>> for name, param in model.named_parameters():
        ...     print(f"{name}: {param.shape}")
    """

def zero_grad(self) -> None:
    """
    Zero out gradients of all parameters.
    
    Example:
        >>> model.zero_grad()  # Clear all gradients
    """
```

##### Module Hierarchy
```python
def add_module(self, name: str, module: Optional[Module]) -> None:
    """
    Add a child module.
    
    Args:
        name: Name for the submodule
        module: Module instance to add
        
    Example:
        >>> model = nn.Module()
        >>> model.add_module('fc', nn.Linear(10, 5))
    """

def modules(self) -> Iterator[Module]:
    """Return iterator over all modules (including self)."""

def children(self) -> Iterator[Module]:
    """Return iterator over immediate child modules."""

def named_modules(self) -> Iterator[Tuple[str, Module]]:
    """Return iterator over all modules with names."""
```

##### Training Mode
```python
def train(self, mode: bool = True) -> Module:
    """
    Set module to training mode.
    
    Args:
        mode: Whether to enable training mode
        
    Returns:
        self
        
    Example:
        >>> model.train()  # Enable training mode
        >>> model.train(False)  # Equivalent to model.eval()
    """

def eval(self) -> Module:
    """
    Set module to evaluation mode.
    
    Returns:
        self
        
    Example:
        >>> model.eval()  # Disable dropout, use running stats for BN
    """
```

##### State Management
```python
def state_dict(self) -> Dict[str, Tensor]:
    """
    Return state dictionary containing all parameters and buffers.
    
    Returns:
        Dictionary mapping parameter names to tensors
        
    Example:
        >>> state = model.state_dict()
        >>> genesis.save(state, 'model.pth')
    """

def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
    """
    Load parameters from state dictionary.
    
    Args:
        state_dict: Dictionary of parameters
        
    Example:
        >>> state = genesis.load('model.pth')
        >>> model.load_state_dict(state)
    """
```

### `nn.Parameter`

A special tensor that is automatically registered as a module parameter.

```python
class Parameter(Tensor):
    """
    A tensor that is automatically registered as a module parameter.
    
    Args:
        data: Tensor data
        requires_grad: Whether to compute gradients (default: True)
        
    Example:
        >>> class MyModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = nn.Parameter(genesis.randn(10, 5))
    """
```

## Layer Types

### Linear Layers

#### `nn.Linear`

Fully connected layer performing linear transformation.

```python
class Linear(Module):
    """
    Linear transformation: y = xW^T + b
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term (default: True)
        
    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)
        
    Example:
        >>> layer = nn.Linear(20, 30)
        >>> x = genesis.randn(128, 20)
        >>> output = layer(x)  # Shape: (128, 30)
    """
```

### Convolutional Layers

#### `nn.Conv2d`

2D convolution layer for image processing.

```python
class Conv2d(Module):
    """
    2D convolution over input signal.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution (default: 1)
        padding: Zero-padding added to both sides (default: 0)
        bias: Whether to add bias (default: True)
        
    Shape:
        - Input: (N, C_in, H, W)
        - Output: (N, C_out, H_out, W_out)
        
    Example:
        >>> conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        >>> x = genesis.randn(32, 3, 224, 224)
        >>> output = conv(x)  # Shape: (32, 64, 224, 224)
    """
```

### Activation Functions

#### `nn.ReLU`

Rectified Linear Unit activation.

```python
class ReLU(Module):
    """
    ReLU activation: f(x) = max(0, x)
    
    Args:
        inplace: Whether to modify input in-place (default: False)
        
    Example:
        >>> relu = nn.ReLU()
        >>> x = genesis.randn(10)
        >>> output = relu(x)
    """
```

#### `nn.Sigmoid`

Sigmoid activation function.

```python
class Sigmoid(Module):
    """
    Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    
    Example:
        >>> sigmoid = nn.Sigmoid()
        >>> x = genesis.randn(10)
        >>> output = sigmoid(x)  # Values in (0, 1)
    """
```

#### `nn.Tanh`

Hyperbolic tangent activation.

```python
class Tanh(Module):
    """
    Tanh activation: f(x) = tanh(x)
    
    Example:
        >>> tanh = nn.Tanh()
        >>> x = genesis.randn(10)
        >>> output = tanh(x)  # Values in (-1, 1)
    """
```

#### `nn.SiLU` (Swish)

Sigmoid Linear Unit activation.

```python
class SiLU(Module):
    """
    SiLU/Swish activation: f(x) = x * sigmoid(x)
    
    Example:
        >>> silu = nn.SiLU()
        >>> x = genesis.randn(10)
        >>> output = silu(x)
    """
```

#### `nn.GELU`

Gaussian Error Linear Unit activation.

```python
class GELU(Module):
    """
    GELU activation: f(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of standard Gaussian.
    
    Example:
        >>> gelu = nn.GELU()
        >>> x = genesis.randn(10)
        >>> output = gelu(x)
    """
```

#### `nn.Softmax`

Softmax activation for multi-class classification.

```python
class Softmax(Module):
    """
    Softmax activation: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    
    Args:
        dim: Dimension along which to apply softmax
        
    Example:
        >>> softmax = nn.Softmax(dim=-1)
        >>> x = genesis.randn(10, 5)
        >>> output = softmax(x)  # Each row sums to 1
    """
```

### Normalization Layers

#### `nn.BatchNorm1d`

Batch normalization for 1D or 2D inputs.

```python
class BatchNorm1d(Module):
    """
    Batch normalization over 2D or 3D input.
    
    Args:
        num_features: Number of features (C in [N, C] or [N, C, L])
        eps: Small value for numerical stability (default: 1e-5)
        momentum: Momentum for running stats (default: 0.1)
        
    Shape:
        - Input: (N, C) or (N, C, L)
        - Output: Same as input
        
    Example:
        >>> bn = nn.BatchNorm1d(100)
        >>> x = genesis.randn(20, 100)
        >>> output = bn(x)
    """
```

#### `nn.LayerNorm`

Layer normalization.

```python
class LayerNorm(Module):
    """
    Layer normalization over last dimensions.
    
    Args:
        normalized_shape: Shape of dimensions to normalize
        eps: Small value for numerical stability (default: 1e-5)
        
    Shape:
        - Input: (*, normalized_shape)
        - Output: Same as input
        
    Example:
        >>> ln = nn.LayerNorm([768])
        >>> x = genesis.randn(32, 100, 768)
        >>> output = ln(x)  # Normalize over last dimension
    """
```

### Dropout Layers

#### `nn.Dropout`

Dropout for regularization.

```python
class Dropout(Module):
    """
    Randomly zero out elements for regularization.
    
    Args:
        p: Probability of zeroing an element (default: 0.5)
        inplace: Whether to modify input in-place (default: False)
        
    Example:
        >>> dropout = nn.Dropout(p=0.2)
        >>> x = genesis.randn(20, 16)
        >>> output = dropout(x)  # Training mode: randomly zero 20% of elements
    """
```

### Pooling Layers

#### `nn.MaxPool2d`

2D max pooling.

```python
class MaxPool2d(Module):
    """
    Max pooling over 2D input.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Zero-padding (default: 0)
        
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H_out, W_out)
        
    Example:
        >>> pool = nn.MaxPool2d(kernel_size=2, stride=2)
        >>> x = genesis.randn(1, 16, 32, 32)
        >>> output = pool(x)  # Shape: (1, 16, 16, 16)
    """
```

#### `nn.AvgPool2d`

2D average pooling.

```python
class AvgPool2d(Module):
    """
    Average pooling over 2D input.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Zero-padding (default: 0)
        
    Example:
        >>> pool = nn.AvgPool2d(kernel_size=2, stride=2)
        >>> x = genesis.randn(1, 16, 32, 32)
        >>> output = pool(x)  # Shape: (1, 16, 16, 16)
    """
```

### Embedding Layers

#### `nn.Embedding`

Embedding lookup table.

```python
class Embedding(Module):
    """
    Embedding lookup table.
    
    Args:
        num_embeddings: Size of vocabulary
        embedding_dim: Dimension of embeddings
        
    Shape:
        - Input: (*) containing indices
        - Output: (*, embedding_dim)
        
    Example:
        >>> embed = nn.Embedding(10000, 300)  # 10k vocab, 300-dim embeddings
        >>> indices = genesis.tensor([1, 2, 3, 4])
        >>> output = embed(indices)  # Shape: (4, 300)
    """
```

### Attention Layers

#### `nn.MultiheadAttention`

Multi-head attention mechanism.

```python
class MultiheadAttention(Module):
    """
    Multi-head attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to add bias (default: True)
        
    Shape:
        - Query: (L, N, E) or (N, L, E)
        - Key: (S, N, E) or (N, S, E)
        - Value: (S, N, E) or (N, S, E)
        - Output: (L, N, E) or (N, L, E)
        
    Example:
        >>> attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        >>> x = genesis.randn(10, 32, 512)  # (seq_len, batch, embed_dim)
        >>> output, weights = attn(x, x, x)
    """
```

## Container Modules

### `nn.Sequential`

Sequential container for modules.

```python
class Sequential(Module):
    """
    Sequential container that runs modules in order.
    
    Args:
        *modules: Sequence of modules to apply
        
    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> x = genesis.randn(32, 784)
        >>> output = model(x)  # Shape: (32, 10)
    """
```

### `nn.ModuleList`

List container for modules.

```python
class ModuleList(Module):
    """
    List of modules that are properly registered.
    
    Args:
        modules: Optional list of modules
        
    Example:
        >>> layers = nn.ModuleList([
        ...     nn.Linear(10, 10) for _ in range(5)
        ... ])
        >>> x = genesis.randn(32, 10)
        >>> for layer in layers:
        ...     x = layer(x)
    """
```

### `nn.ModuleDict`

Dictionary container for modules.

```python
class ModuleDict(Module):
    """
    Dictionary of modules with string keys.
    
    Args:
        modules: Optional dict of modules
        
    Example:
        >>> layers = nn.ModuleDict({
        ...     'fc1': nn.Linear(10, 20),
        ...     'fc2': nn.Linear(20, 10)
        ... })
        >>> x = genesis.randn(32, 10)
        >>> x = layers['fc1'](x)
        >>> x = layers['fc2'](x)
    """
```

## Loss Functions

### `nn.MSELoss`

Mean squared error loss.

```python
class MSELoss(Module):
    """
    Mean squared error loss: L = mean((y_pred - y_true)^2)
    
    Args:
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
        
    Example:
        >>> loss_fn = nn.MSELoss()
        >>> pred = genesis.randn(32, 10)
        >>> target = genesis.randn(32, 10)
        >>> loss = loss_fn(pred, target)
    """
```

### `nn.CrossEntropyLoss`

Cross entropy loss for classification.

```python
class CrossEntropyLoss(Module):
    """
    Cross entropy loss for multi-class classification.
    
    Args:
        weight: Manual rescaling weight for each class
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
        
    Shape:
        - Input: (N, C) where C is number of classes
        - Target: (N,) containing class indices
        
    Example:
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> logits = genesis.randn(32, 10)  # 32 samples, 10 classes
        >>> targets = genesis.randint(0, 10, (32,))
        >>> loss = loss_fn(logits, targets)
    """
```

### `nn.BCELoss`

Binary cross entropy loss.

```python
class BCELoss(Module):
    """
    Binary cross entropy loss.
    
    Args:
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
        
    Shape:
        - Input: (N, *) where * means any number of dimensions
        - Target: Same shape as input
        
    Example:
        >>> loss_fn = nn.BCELoss()
        >>> pred = genesis.sigmoid(genesis.randn(32, 1))
        >>> target = genesis.randint(0, 2, (32, 1)).float()
        >>> loss = loss_fn(pred, target)
    """
```

## Utilities

### Weight Initialization

```python
def init_weights(module: Module, init_type: str = 'xavier'):
    """
    Initialize module weights.
    
    Args:
        module: Module to initialize
        init_type: 'xavier', 'kaiming', 'normal', 'uniform'
        
    Example:
        >>> model = nn.Linear(10, 5)
        >>> init_weights(model, 'xavier')
    """
```

### Gradient Clipping

```python
def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0):
    """
    Clip gradients by norm.
    
    Args:
        parameters: Iterable of parameters
        max_norm: Maximum norm value
        norm_type: Type of norm (default: 2.0)
        
    Example:
        >>> nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    """

def clip_grad_value_(parameters, clip_value: float):
    """
    Clip gradients by value.
    
    Args:
        parameters: Iterable of parameters
        clip_value: Maximum absolute value
        
    Example:
        >>> nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
    """
```

## Building Custom Modules

### Example: Custom Layer

```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Parameters are automatically tracked
        self.weight = nn.Parameter(genesis.randn(out_features, in_features))
        self.bias = nn.Parameter(genesis.zeros(out_features))
        
        # Submodules are automatically tracked
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Define forward pass
        x = genesis.matmul(x, self.weight.T) + self.bias
        x = self.activation(x)
        return x

# Usage
layer = CustomLayer(10, 5)
x = genesis.randn(32, 10)
output = layer(x)
```

### Example: Custom Model

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual  # Skip connection
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = nn.Sequential(*[ResidualBlock(64) for _ in range(3)])
        self.layer2 = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## Best Practices

1. **Always override `forward()`**: Define computation in forward method
2. **Use `module(input)`**: Never call forward() directly
3. **Register parameters**: Use nn.Parameter for learnable parameters
4. **Track submodules**: Assign modules as attributes for automatic tracking
5. **Handle training/eval**: Use different behavior for training vs evaluation
6. **Initialize weights**: Proper initialization improves convergence

## See Also

- [Functional API](functional.md) - Functional operations
- [Optimizers](../optim/optimizers.md) - Training optimizers
- [Autograd](../autograd.md) - Automatic differentiation
- [Examples](../../../samples/) - Complete examples