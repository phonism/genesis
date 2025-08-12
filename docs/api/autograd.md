# Automatic Differentiation System (genesis.autograd)

## Overview

The automatic differentiation system is the core of Genesis, providing dynamic computational graph construction and automatic gradient computation. It implements reverse-mode automatic differentiation (backpropagation) with support for complex computational graphs.

## Core Concepts

### Computational Graph

Genesis builds a dynamic computational graph as operations are performed. Each operation creates nodes in the graph that track:
- Input tensors
- The operation performed
- Output tensors
- Gradient functions for backpropagation

### Gradient Computation

Gradients are computed using the chain rule, traversing the computational graph in reverse order from outputs to inputs.

## Main Classes

### `genesis.Tensor`

The fundamental data structure in Genesis that supports automatic differentiation.

```python
class Tensor:
    def __init__(
        self,
        array: Union[list, np.ndarray, NDArray],
        device: Optional[Device] = None,
        dtype: Optional[DType] = None,
        requires_grad: bool = False,
        **kwargs
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | array-like | required | Input data (list, numpy array, or NDArray) |
| `device` | Device | `None` | Computation device (cpu/cuda) |
| `dtype` | DType | `None` | Data type (inferred if None) |
| `requires_grad` | bool | `False` | Whether to compute gradients |
| `**kwargs` | dict | `{}` | Additional NDArray parameters |

#### Properties

##### Shape and Type Information
```python
@property
def shape(self) -> Tuple[int, ...]:
    """Returns the shape of the tensor."""

@property
def dtype(self) -> DType:
    """Returns the data type."""

@property
def device(self) -> Device:
    """Returns the device."""

@property
def ndim(self) -> int:
    """Returns the number of dimensions."""

@property
def size(self) -> int:
    """Returns the total number of elements."""
```

##### Gradient Properties
```python
@property
def requires_grad(self) -> bool:
    """Whether this tensor requires gradient computation."""

@property
def grad(self) -> Optional[Tensor]:
    """Access the gradient tensor."""

@property
def is_leaf(self) -> bool:
    """Whether this is a leaf node (user-created tensor)."""

@property
def grad_fn(self) -> Optional[Function]:
    """The function that created this tensor."""
```

#### Core Methods

##### Gradient Operations
```python
def backward(self, gradient: Optional[Tensor] = None) -> None:
    """
    Compute gradients via backpropagation.
    
    Args:
        gradient: Output gradient. Defaults to tensor([1.0]) for scalars.
        
    Example:
        >>> x = genesis.tensor([1., 2., 3.], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> y.backward()
        >>> print(x.grad)  # tensor([2., 4., 6.])
    """

def detach(self) -> Tensor:
    """
    Returns a new tensor detached from the computational graph.
    
    Returns:
        Tensor with requires_grad=False
        
    Example:
        >>> x = genesis.tensor([1., 2.], requires_grad=True)
        >>> y = x.detach()
        >>> print(y.requires_grad)  # False
    """

def retain_grad(self) -> None:
    """
    Enable gradient retention for non-leaf tensors.
    
    Example:
        >>> x = genesis.tensor([1., 2.], requires_grad=True)
        >>> y = x * 2  # Non-leaf tensor
        >>> y.retain_grad()
        >>> z = y.sum()
        >>> z.backward()
        >>> print(y.grad)  # tensor([1., 1.])
    """

def zero_grad(self) -> None:
    """
    Zero out the gradient tensor.
    
    Example:
        >>> x = genesis.tensor([1., 2.], requires_grad=True)
        >>> y = x.sum()
        >>> y.backward()
        >>> x.zero_grad()
        >>> print(x.grad)  # None
    """
```

##### Tensor Operations

All standard mathematical operations are supported and automatically tracked for gradient computation:

```python
# Arithmetic operations
z = x + y          # Addition
z = x - y          # Subtraction
z = x * y          # Multiplication
z = x / y          # Division
z = x ** y         # Power
z = x @ y          # Matrix multiplication

# Unary operations
z = -x             # Negation
z = x.abs()        # Absolute value
z = x.exp()        # Exponential
z = x.log()        # Natural logarithm
z = x.sqrt()       # Square root
z = x.sin()        # Sine
z = x.cos()        # Cosine
z = x.tanh()       # Hyperbolic tangent

# Reduction operations
z = x.sum()        # Sum all elements
z = x.mean()       # Mean of all elements
z = x.max()        # Maximum element
z = x.min()        # Minimum element

# Shape operations
z = x.reshape(shape)      # Reshape
z = x.transpose(dims)     # Transpose
z = x.squeeze()           # Remove singleton dimensions
z = x.unsqueeze(dim)      # Add singleton dimension
z = x.view(shape)         # View with different shape
```

### `genesis.Function`

Base class for all differentiable operations.

```python
class Function:
    """
    Base class for implementing custom differentiable operations.
    """
    
    @staticmethod
    def forward(ctx: Context, *args, **kwargs) -> Tensor:
        """
        Forward pass implementation.
        
        Args:
            ctx: Context object for saving information for backward pass
            *args: Input tensors
            **kwargs: Additional arguments
            
        Returns:
            Output tensor(s)
        """
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass implementation.
        
        Args:
            ctx: Context object with saved information
            *grad_outputs: Gradients w.r.t. outputs
            
        Returns:
            Gradients w.r.t. inputs (None for non-differentiable inputs)
        """
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        """
        Apply the function and register it in the computational graph.
        """
```

#### Custom Function Example

```python
import genesis
from genesis import Function

class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for backward pass
        ctx.save_for_backward(x)
        return genesis.tensor(x.data.exp(), requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensor
        x, = ctx.saved_tensors
        # Gradient of exp(x) is exp(x)
        return grad_output * x.exp()

# Usage
exp = Exp.apply
x = genesis.tensor([1., 2., 3.], requires_grad=True)
y = exp(x)
y.sum().backward()
print(x.grad)  # Gradients computed through custom function
```

## Context Management

### `genesis.no_grad()`

Context manager to disable gradient computation for efficiency during inference.

```python
with genesis.no_grad():
    # Operations here won't build computational graph
    y = model(x)  # No gradients computed
```

### `genesis.enable_grad()`

Context manager to enable gradient computation (useful within no_grad context).

```python
with genesis.no_grad():
    # Most operations without gradients
    y = model(x)
    
    with genesis.enable_grad():
        # This specific operation needs gradients
        z = y.sum()
        z.backward()
```

### `genesis.set_grad_enabled(mode: bool)`

Globally enable or disable gradient computation.

```python
genesis.set_grad_enabled(False)  # Disable globally
y = x * 2  # No gradients

genesis.set_grad_enabled(True)   # Enable globally
z = x * 2  # Gradients computed
```

## Gradient Hooks

### Pre and Post Hooks

Register functions to be called during backward pass:

```python
def print_grad(grad):
    print(f"Gradient: {grad}")
    return grad  # Can modify gradient here

x = genesis.tensor([1., 2., 3.], requires_grad=True)
x.register_hook(print_grad)
y = (x ** 2).sum()
y.backward()  # Will print gradients during backward
```

## Memory Management

### Gradient Accumulation

Gradients accumulate by default across multiple backward passes:

```python
x = genesis.tensor([1., 2.], requires_grad=True)

y1 = x.sum()
y1.backward()
print(x.grad)  # tensor([1., 1.])

y2 = (x * 2).sum()
y2.backward()
print(x.grad)  # tensor([3., 3.]) - accumulated!
```

### Clearing Gradients

```python
# Clear gradients before new computation
x.grad = None  # or x.zero_grad()
```

## Best Practices

### 1. Efficient Inference

Always use `no_grad()` context for inference:

```python
model.eval()
with genesis.no_grad():
    predictions = model(test_data)
```

### 2. Memory Optimization

Detach intermediate results when gradients aren't needed:

```python
# Don't need gradients for running_mean
running_mean = (alpha * running_mean.detach() + 
                (1 - alpha) * batch_mean)
```

### 3. Gradient Clipping

Prevent gradient explosion:

```python
genesis.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Mixed Precision Training

Use automatic mixed precision for faster training:

```python
genesis.enable_autocast = True
with genesis.autocast():
    output = model(input)
    loss = criterion(output, target)
```

## Common Patterns

### Training Loop

```python
model = MyModel()
optimizer = genesis.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch.inputs)
        loss = criterion(outputs, batch.targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
```

### Gradient Checkpointing

Save memory by recomputing activations:

```python
# Coming in future versions
from genesis.utils.checkpoint import checkpoint

def forward(self, x):
    # Checkpoint intermediate computation
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return self.layer3(x)
```

## Debugging

### Gradient Checking

Verify gradients with numerical differentiation:

```python
from genesis.autograd import gradcheck

def func(x):
    return (x ** 2).sum()

x = genesis.tensor([1., 2., 3.], requires_grad=True)
gradcheck(func, x, eps=1e-6)  # Returns True if gradients are correct
```

### Inspecting Computational Graph

```python
# Print computational graph structure
y = x * 2 + 3
print(y.grad_fn)  # <AddBackward>
print(y.grad_fn.next_functions)  # Connected operations
```

## Performance Tips

1. **Reuse tensors**: Avoid creating new tensors unnecessarily
2. **In-place operations**: Use when possible (e.g., `x.add_(y)`)
3. **Batch operations**: Process multiple samples together
4. **Disable gradients**: Use `no_grad()` for inference
5. **Clear gradients**: Zero gradients before each backward pass

## See Also

- [Neural Network Modules](nn/modules.md) - Building models with Genesis
- [Optimizers](optim/optimizers.md) - Training with gradient descent
- [Tensor Operations](../ndarray/index.md) - Low-level tensor operations
- [Examples](../../samples/) - Complete working examples