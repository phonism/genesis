# Tensor API Reference

The Tensor class is the fundamental data structure in Genesis, providing automatic differentiation support and efficient computation on CPU and GPU devices.

## Tensor Class

```python
genesis.tensor(data, dtype=None, device=None, requires_grad=False)
genesis.Tensor(data, dtype=None, device=None, requires_grad=False)
```

### Parameters

- **data**: Array-like data (list, numpy array, or existing tensor)
- **dtype**: Data type (default: inferred from data)
  - `genesis.float32`, `genesis.float64`, `genesis.float16`, `genesis.bfloat16`
  - `genesis.int32`, `genesis.int64`, `genesis.int16`, `genesis.int8`
  - `genesis.uint8`, `genesis.bool`
- **device**: Device to place tensor on
  - `genesis.device('cpu')`: CPU device
  - `genesis.device('cuda')`: Default CUDA device
  - `genesis.device('cuda:0')`: Specific CUDA device
- **requires_grad**: Whether to track gradients for automatic differentiation

### Properties

- **shape**: Tuple of tensor dimensions
- **dtype**: Data type of the tensor
- **device**: Device where tensor is stored
- **requires_grad**: Whether tensor requires gradient computation
- **grad**: Gradient tensor (after backward pass)
- **data**: Underlying storage without gradient tracking

### Methods

#### Basic Operations

```python
# Element-wise operations
tensor.add(other)        # Addition
tensor.sub(other)        # Subtraction
tensor.mul(other)        # Multiplication
tensor.div(other)        # Division
tensor.pow(exponent)     # Power
tensor.exp()            # Exponential
tensor.log()            # Natural logarithm
tensor.sqrt()           # Square root
tensor.abs()            # Absolute value
tensor.neg()            # Negation

# Reduction operations
tensor.sum(axis=None, keepdims=False)      # Sum
tensor.mean(axis=None, keepdims=False)     # Mean
tensor.max(axis=None, keepdims=False)      # Maximum
tensor.min(axis=None, keepdims=False)      # Minimum

# Shape operations
tensor.reshape(shape)                       # Reshape
tensor.transpose(axes=None)                # Transpose
tensor.squeeze(axis=None)                  # Remove dimensions of size 1
tensor.unsqueeze(axis)                     # Add dimension of size 1
tensor.flatten(start_dim=0, end_dim=-1)    # Flatten dimensions
```

#### Matrix Operations

```python
# Matrix multiplication
tensor.matmul(other)     # Matrix multiplication
tensor @ other           # Matrix multiplication operator

# Linear algebra
tensor.T                 # Transpose (2D tensors)
```

#### Gradient Operations

```python
# Automatic differentiation
tensor.backward()        # Compute gradients
tensor.detach()         # Detach from computation graph
tensor.zero_grad()      # Zero out gradients
```

#### Device Operations

```python
# Move between devices
tensor.to(device)       # Move to specified device
tensor.cpu()           # Move to CPU
tensor.cuda()          # Move to CUDA
```

#### Type Conversion

```python
# Data type conversion
tensor.float()         # Convert to float32
tensor.double()        # Convert to float64
tensor.half()          # Convert to float16
tensor.int()           # Convert to int32
tensor.long()          # Convert to int64
tensor.bool()          # Convert to bool

# Convert to other formats
tensor.numpy()         # Convert to numpy array
tensor.tolist()        # Convert to Python list
```

## Examples

### Basic Tensor Creation

```python
import genesis

# From list
x = genesis.tensor([1, 2, 3, 4])

# From numpy array
import numpy as np
arr = np.array([[1, 2], [3, 4]])
y = genesis.tensor(arr, dtype=genesis.float32)

# On GPU
z = genesis.tensor([1, 2, 3], device=genesis.device('cuda'))

# With gradient tracking
w = genesis.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

### Automatic Differentiation

```python
import genesis

# Create tensors with gradient tracking
x = genesis.tensor([2.0], requires_grad=True)
y = genesis.tensor([3.0], requires_grad=True)

# Forward pass
z = x * y + x ** 2
print(f"z = {z}")  # z = [10.0]

# Backward pass
z.backward()
print(f"x.grad = {x.grad}")  # x.grad = [7.0] (dy/dx = y + 2x = 3 + 4 = 7)
print(f"y.grad = {y.grad}")  # y.grad = [2.0] (dy/dy = x = 2)
```

### Matrix Operations

```python
import genesis

# Create matrices
A = genesis.randn(3, 4)
B = genesis.randn(4, 5)

# Matrix multiplication
C = A @ B  # or A.matmul(B)
print(f"Result shape: {C.shape}")  # (3, 5)

# Element-wise operations
D = A * 2 + 1
E = A.exp().sum(axis=1, keepdims=True)
```

### Device Management

```python
import genesis

# Check CUDA availability
if genesis.cuda_available():
    device = genesis.device('cuda')
else:
    device = genesis.device('cpu')

# Create tensor on specific device
x = genesis.randn(100, 100, device=device)

# Move tensor between devices
x_cpu = x.cpu()
x_cuda = x_cpu.cuda()
```

## See Also

- [Functional API](functional.md) - Functional operations
- [Neural Networks](nn.md) - Neural network modules
- [Device Management](device.md) - Device abstraction and management
- [Storage Layer](storage.md) - Memory storage interface