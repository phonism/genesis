# Functional Operations Interface (genesis.nn.functional)

Genesis functional interface provides stateless tensor operation functions that can be called directly on tensors without creating module instances.

## Module Overview

`genesis.nn.functional` (commonly imported as `F`) includes:
- **Basic arithmetic operations** (add, subtract, multiply, divide)
- **Mathematical functions** (sin, cos, log, exp, sqrt, power)
- **Tensor shape operations** (transpose, reshape, expand, view, flatten)
- **Tensor indexing and slicing** (getitem, setitem, broadcast_to)
- **Aggregation operations** (sum, max, logsumexp)
- **Matrix operations** (matmul, stack, cat, squeeze, unsqueeze)
- **Basic activation functions** (relu)
- **Advanced operations** (softmax, dropout from triton_ops)

## Basic Arithmetic Operations

### add
```python
def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise addition of two tensors.
    
    Args:
        a: Tensor - First input tensor
        b: Tensor - Second input tensor
        
    Returns:
        Tensor - Element-wise sum a + b
        
    Example:
        >>> x = genesis.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = genesis.tensor([[2.0, 1.0], [1.0, 2.0]])
        >>> z = F.add(x, y)
        >>> # Result: [[3.0, 3.0], [4.0, 6.0]]
    """
```

### sub
```python
def sub(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise subtraction of two tensors.
    
    Args:
        a: Tensor - First input tensor (minuend)
        b: Tensor - Second input tensor (subtrahend)
        
    Returns:
        Tensor - Element-wise difference a - b
        
    Example:
        >>> x = genesis.tensor([5.0, 3.0, 8.0])
        >>> y = genesis.tensor([2.0, 1.0, 3.0])
        >>> z = F.sub(x, y)
        >>> # Result: [3.0, 2.0, 5.0]
    """
```

### multiply
```python
def multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise multiplication of two tensors.
    
    Args:
        a: Tensor - First input tensor
        b: Tensor - Second input tensor
        
    Returns:
        Tensor - Element-wise product a * b
        
    Example:
        >>> x = genesis.tensor([2.0, 3.0, 4.0])
        >>> y = genesis.tensor([1.5, 2.0, 0.5])
        >>> z = F.multiply(x, y)
        >>> # Result: [3.0, 6.0, 2.0]
    """
```

### divide
```python
def divide(a: Tensor, b: Tensor) -> Tensor:
    """
    Element-wise division of two tensors.
    
    Args:
        a: Tensor - Dividend tensor
        b: Tensor - Divisor tensor
        
    Returns:
        Tensor - Element-wise quotient a / b
        
    Example:
        >>> x = genesis.tensor([6.0, 8.0, 9.0])
        >>> y = genesis.tensor([2.0, 4.0, 3.0])
        >>> z = F.divide(x, y)
        >>> # Result: [3.0, 2.0, 3.0]
    """
```

## Scalar Operations

### add_scalar, mul_scalar, divide_scalar, pow_scalar
```python
def add_scalar(a: Tensor, scalar: float) -> Tensor:
def mul_scalar(a: Tensor, scalar: float) -> Tensor:
def divide_scalar(a: Tensor, scalar: float, reverse: bool = False) -> Tensor:
def pow_scalar(a: Tensor, scalar: float, reverse: bool = False) -> Tensor:
    """
    Element-wise operations between tensor and scalar.
    
    Args:
        a: Tensor - Input tensor
        scalar: float - Scalar value
        reverse: bool - If True, applies scalar op tensor (for divide/pow)
        
    Returns:
        Tensor - Result of tensor-scalar operation
        
    Example:
        >>> x = genesis.tensor([1.0, 2.0, 3.0])
        >>> y1 = F.add_scalar(x, 5.0)      # [6.0, 7.0, 8.0]
        >>> y2 = F.mul_scalar(x, 2.0)      # [2.0, 4.0, 6.0]
        >>> y3 = F.pow_scalar(x, 2.0)      # [1.0, 4.0, 9.0]
    """
```

## Mathematical Functions

### sin, cos, log, exp, sqrt
```python
def sin(a: Tensor) -> Tensor:
def cos(a: Tensor) -> Tensor:
def log(a: Tensor) -> Tensor:
def exp(a: Tensor) -> Tensor:
def sqrt(a: Tensor) -> Tensor:
    """
    Element-wise mathematical functions.
    
    Args:
        a: Tensor - Input tensor
        
    Returns:
        Tensor - Result of mathematical function
        
    Example:
        >>> x = genesis.tensor([0.0, 1.0, 2.0])
        >>> y1 = F.sin(x)   # [0.0, 0.841, 0.909]
        >>> y2 = F.exp(x)   # [1.0, 2.718, 7.389]
        >>> y3 = F.sqrt(genesis.tensor([4.0, 9.0, 16.0]))  # [2.0, 3.0, 4.0]
    """
```

### negate
```python
def negate(a: Tensor) -> Tensor:
    """
    Element-wise negation: -a
    
    Args:
        a: Tensor - Input tensor
        
    Returns:
        Tensor - Negated tensor
        
    Example:
        >>> x = genesis.tensor([1.0, -2.0, 3.0])
        >>> y = F.negate(x)
        >>> # Result: [-1.0, 2.0, -3.0]
    """
```

## Shape Operations

### transpose
```python
def transpose(a: Tensor, axis: tuple = None) -> Tensor:
    """
    Transpose tensor dimensions.
    
    Args:
        a: Tensor - Input tensor
        axis: tuple - Pair of dimensions to swap (default: last two dims)
        
    Returns:
        Tensor - Transposed tensor
        
    Example:
        >>> x = genesis.randn(3, 4, 5)
        >>> y1 = F.transpose(x)           # Swap last two dims: (3, 5, 4)
        >>> y2 = F.transpose(x, (0, 2))   # Swap dims 0,2: (5, 4, 3)
    """
```

### reshape
```python
def reshape(a: Tensor, shape: tuple) -> Tensor:
    """
    Reshape tensor to new shape.
    
    Args:
        a: Tensor - Input tensor
        shape: tuple - New shape (must have same total elements)
        
    Returns:
        Tensor - Reshaped tensor
        
    Example:
        >>> x = genesis.randn(2, 6)
        >>> y = F.reshape(x, (3, 4))
        >>> # Changes shape from (2, 6) to (3, 4)
    """
```

### view, expand, flatten
```python
def view(a: Tensor, shape: tuple) -> Tensor:
def expand(a: Tensor, shape: tuple) -> Tensor:
def flatten(a: Tensor, start_dim: int = 0, end_dim: int = None) -> Tensor:
    """
    Tensor view and shape manipulation operations.
    
    Args:
        a: Tensor - Input tensor
        shape: tuple - Target shape
        start_dim, end_dim: int - Dimensions to flatten
        
    Returns:
        Tensor - Transformed tensor
        
    Example:
        >>> x = genesis.randn(2, 3, 4)
        >>> y1 = F.view(x, (6, 4))         # View as (6, 4)
        >>> y2 = F.expand(x, (2, 3, 4, 5)) # Expand last dim
        >>> y3 = F.flatten(x, 1)           # Flatten from dim 1: (2, 12)
    """
```

## Tensor Operations

### matmul
```python
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication.
    
    Args:
        a: Tensor - Left matrix
        b: Tensor - Right matrix
        
    Returns:
        Tensor - Matrix product
        
    Example:
        >>> x = genesis.randn(3, 4)
        >>> y = genesis.randn(4, 5)
        >>> z = F.matmul(x, y)  # Shape: (3, 5)
    """
```

### stack, cat
```python
def stack(tensors: list, dim: int = 0) -> Tensor:
def cat(tensors: list, dim: int = 0) -> Tensor:
    """
    Stack or concatenate tensors along dimension.
    
    Args:
        tensors: list - List of tensors to combine
        dim: int - Dimension along which to stack/concatenate
        
    Returns:
        Tensor - Combined tensor
        
    Example:
        >>> x = genesis.randn(2, 3)
        >>> y = genesis.randn(2, 3)
        >>> z1 = F.stack([x, y], dim=0)  # Shape: (2, 2, 3)
        >>> z2 = F.cat([x, y], dim=0)    # Shape: (4, 3)
    """
```

### squeeze, unsqueeze
```python
def squeeze(tensor: Tensor, dim: int) -> Tensor:
def unsqueeze(tensor: Tensor, dim: int) -> Tensor:
    """
    Remove or add singleton dimensions.
    
    Args:
        tensor: Tensor - Input tensor
        dim: int - Dimension to squeeze/unsqueeze
        
    Returns:
        Tensor - Tensor with modified dimensions
        
    Example:
        >>> x = genesis.randn(1, 3, 1, 4)
        >>> y1 = F.squeeze(x, 0)    # Shape: (3, 1, 4)
        >>> y2 = F.unsqueeze(x, 2)  # Shape: (1, 3, 1, 1, 4)
    """
```

## Aggregation Operations

### sum
```python
def sum(a: Tensor, axis: int = None, keepdims: bool = False) -> Tensor:
    """
    Sum tensor elements along specified dimensions.
    
    Args:
        a: Tensor - Input tensor
        axis: int - Dimension to sum over (None for all)
        keepdims: bool - Whether to keep reduced dimensions
        
    Returns:
        Tensor - Summed tensor
        
    Example:
        >>> x = genesis.randn(3, 4)
        >>> y1 = F.sum(x)           # Sum all elements: scalar
        >>> y2 = F.sum(x, axis=0)   # Sum over rows: shape (4,)
        >>> y3 = F.sum(x, axis=1, keepdims=True)  # Shape: (3, 1)
    """
```

### max, logsumexp
```python
def max(a: Tensor, axis: int = None, keepdims: bool = False) -> Tensor:
def logsumexp(a: Tensor, axis: int = None) -> Tensor:
    """
    Maximum and log-sum-exp operations.
    
    Args:
        a: Tensor - Input tensor
        axis: int - Dimension to reduce over
        keepdims: bool - Whether to keep reduced dimensions
        
    Returns:
        Tensor - Result tensor
        
    Example:
        >>> x = genesis.randn(3, 4)
        >>> y1 = F.max(x, axis=1)      # Max along rows
        >>> y2 = F.logsumexp(x, axis=0) # LogSumExp along cols
    """
```

## Activation Functions

### relu
```python
def relu(a: Tensor) -> Tensor:
    """
    ReLU activation function: f(x) = max(0, x)
    
    Args:
        a: Tensor - Input tensor
        
    Returns:
        Tensor - ReLU-activated tensor
        
    Example:
        >>> x = genesis.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> y = F.relu(x)
        >>> # Result: [0.0, 0.0, 0.0, 1.0, 2.0]
    """
```

## Advanced Operations (from triton_ops)

### softmax
```python
# Imported from genesis.nn.triton_ops
from genesis.nn.triton_ops import softmax

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Softmax function using optimized Triton kernel.
    
    Args:
        x: Tensor - Input tensor
        dim: int - Dimension along which to apply softmax
        
    Returns:
        Tensor - Softmax output (sums to 1 along dim)
        
    Example:
        >>> x = genesis.randn(2, 3)
        >>> y = softmax(x, dim=1)
        >>> # Each row sums to 1
    """
```

### dropout
```python
# Imported from genesis.nn.triton_ops
from genesis.nn.triton_ops import dropout

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Dropout regularization using Triton kernel.
    
    Args:
        x: Tensor - Input tensor
        p: float - Dropout probability
        training: bool - Whether in training mode
        
    Returns:
        Tensor - Tensor with dropout applied
        
    Example:
        >>> x = genesis.randn(100, 50)
        >>> y = dropout(x, p=0.2, training=True)
        >>> # 20% of elements set to 0, others scaled by 1/(1-p)
    """
```

## Indexing and Broadcasting

### getitem, setitem, broadcast_to
```python
def getitem(a: Tensor, index) -> Tensor:
def setitem(a: Tensor, index, value) -> Tensor:
def broadcast_to(a: Tensor, shape: tuple) -> Tensor:
    """
    Tensor indexing and broadcasting operations.
    
    Args:
        a: Tensor - Input tensor
        index: Various - Index (int, slice, list, Tensor)
        value: Tensor/scalar - Value to set
        shape: tuple - Target broadcast shape
        
    Returns:
        Tensor - Indexed/broadcast tensor
        
    Example:
        >>> x = genesis.randn(3, 4)
        >>> y1 = F.getitem(x, [0, 2])      # Select rows 0 and 2
        >>> y2 = F.broadcast_to(x, (2, 3, 4))  # Broadcast to (2, 3, 4)
    """
```

## Performance Notes

- **GPU Acceleration**: Operations automatically use GPU when tensors are on CUDA device
- **Triton Optimization**: Softmax and dropout use optimized Triton kernels
- **Memory Efficiency**: View operations share memory when possible
- **Mixed Precision**: Functions support automatic mixed precision when enabled

## Common Usage Patterns

```python
import genesis
import genesis.nn.functional as F

# Basic operations
x = genesis.randn(100, 784)
y = F.relu(F.matmul(x, weights) + bias)

# Shape manipulation
x = genesis.randn(32, 3, 224, 224)
x_flat = F.flatten(x, start_dim=1)  # (32, 150528)

# Aggregation
logits = genesis.randn(32, 10)
probs = F.softmax(logits, dim=1)
max_vals = F.max(logits, axis=1)

# Advanced indexing
indices = genesis.tensor([0, 2, 4])
selected = F.getitem(x, indices)
```

## Future Features (Roadmap)

The following functions are planned for future releases:
- Advanced activation functions (gelu, silu, swish)
- Loss functions (cross_entropy, mse_loss, l1_loss)
- Normalization functions (layer_norm, batch_norm)
- Convolution operations (conv1d, conv2d)
- Attention mechanisms (scaled_dot_product_attention)

To track progress on these features, see the project roadmap on GitHub.