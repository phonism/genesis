# NDArray System (genesis.ndarray)

## Overview

The `genesis.ndarray` module provides the low-level tensor operations and device abstraction layer that powers Genesis. It implements a dual-backend architecture with optimized operations for both CPU and GPU execution.

## Core Concepts

### Dual Backend Architecture

Genesis uses a unique dual-backend approach:
- **CPU Backend**: Leverages PyTorch for CPU operations with full compatibility
- **GPU Backend**: Pure CUDA/Triton implementation for maximum performance control

### Device Abstraction

All computations are device-agnostic through the `Device` abstraction:
- Automatic device selection and memory management
- Seamless switching between CPU and GPU execution
- Optimized memory allocation patterns

### Performance Optimization

The ndarray system includes several performance optimizations:
- **Kernel Caching**: Compiled Triton kernels are cached for reuse
- **Adaptive Configuration**: Block sizes auto-tune based on tensor dimensions
- **Memory Views**: Efficient tensor views without data copying
- **Broadcast Optimization**: Smart broadcasting for elementwise operations

## Main Classes

### `NDArray`

The fundamental array type in Genesis, providing device-agnostic tensor operations.

```python
class NDArray:
    """
    N-dimensional array with device support.
    
    Args:
        data: Input data (numpy array, list, or tensor)
        device: Target device (cpu or cuda)
        dtype: Data type for the array
        
    Properties:
        shape: Tuple of array dimensions
        dtype: Data type of elements
        device: Device where array is stored
        data: Underlying tensor data
    """
    
    def __init__(
        self, 
        data, 
        device: Optional[Device] = None, 
        dtype: Optional[DType] = None
    ):
```

#### Creation Methods

```python
@staticmethod
def make(
    shape: Tuple[int, ...], 
    device: Optional[Device] = None, 
    dtype: DType = genesis.float32
) -> NDArray:
    """
    Create uninitialized array with specified shape.
    
    Args:
        shape: Dimensions of the array
        device: Target device
        dtype: Element data type
        
    Returns:
        New NDArray instance
        
    Example:
        >>> arr = NDArray.make((10, 20), device=genesis.cuda(), dtype=genesis.float32)
        >>> print(arr.shape)  # (10, 20)
    """
```

#### Properties and Methods

```python
@property
def shape(self) -> Tuple[int, ...]:
    """Array dimensions."""

@property
def dtype(self) -> DType:
    """Element data type."""

@property
def device(self) -> Device:
    """Device where array is stored."""

def numel(self) -> int:
    """
    Total number of elements.
    
    Returns:
        Product of all dimensions
        
    Example:
        >>> arr = NDArray.make((3, 4, 5))
        >>> print(arr.numel())  # 60
    """

def is_contiguous(self) -> bool:
    """
    Check if array has contiguous memory layout.
    
    Returns:
        True if memory is contiguous
    """

def fill(self, value: float) -> None:
    """
    Fill array with constant value in-place.
    
    Args:
        value: Fill value
        
    Example:
        >>> arr = NDArray.make((5, 5))
        >>> arr.fill(0.0)
        >>> # Array now contains all zeros
    """

def numpy(self) -> np.ndarray:
    """
    Convert to NumPy array.
    
    Returns:
        NumPy array with copied data
        
    Example:
        >>> arr = NDArray([1, 2, 3], device=genesis.cuda())
        >>> np_arr = arr.numpy()  # Copies from GPU to CPU
    """

def cpu(self):
    """
    Transfer array to CPU.
    
    Returns:
        CPU version of the array data
    """
```

### `Device`

Abstract device interface supporting CPU and CUDA execution.

```python
class Device:
    """
    Device abstraction for computation backends.
    
    Args:
        name: Device name ('cpu' or 'cuda')
        mod: Backend module for operations
        device_id: GPU device index (for CUDA devices)
    """
    
    def __init__(
        self, 
        name: str, 
        mod: Any, 
        device_id: Optional[int] = None
    ):

    def enabled(self) -> bool:
        """
        Check if device is available.
        
        Returns:
            True if device can be used
        """
```

#### Tensor Creation

```python
def randn(
    self, 
    *shape: int, 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    Create random array from normal distribution.
    
    Args:
        *shape: Array dimensions
        dtype: Element data type
        
    Returns:
        NDArray with random values
        
    Example:
        >>> device = genesis.cuda()
        >>> arr = device.randn(10, 10)  # 10x10 random array
    """

def rand(
    self, 
    *shape: int, 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    Create random array from uniform distribution [0, 1).
    
    Args:
        *shape: Array dimensions
        dtype: Element data type
        
    Returns:
        NDArray with uniform random values
    """

def empty(
    self, 
    shape: Tuple[int, ...], 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    Create uninitialized array.
    
    Args:
        shape: Array dimensions
        dtype: Element data type
        
    Returns:
        Uninitialized NDArray
    """

def full(
    self, 
    shape: Tuple[int, ...], 
    fill_value: float, 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    Create array filled with specified value.
    
    Args:
        shape: Array dimensions
        fill_value: Value to fill array with
        dtype: Element data type
        
    Returns:
        NDArray filled with fill_value
        
    Example:
        >>> device = genesis.cpu()
        >>> ones = device.full((5, 5), 1.0)  # 5x5 array of ones
    """

def one_hot(
    self, 
    n: int, 
    i: NDArray, 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    Create one-hot encoded array.
    
    Args:
        n: Number of classes
        i: Index array
        dtype: Element data type
        
    Returns:
        One-hot encoded NDArray
        
    Example:
        >>> device = genesis.cpu()
        >>> indices = NDArray([0, 2, 1], device=device)
        >>> one_hot = device.one_hot(3, indices)
        >>> # Shape: (3, 3) with one-hot encoding
    """
```

## Device Functions

### Device Creation

```python
def cpu() -> Device:
    """
    Create CPU device.
    
    Returns:
        CPU device instance
        
    Example:
        >>> cpu_device = genesis.cpu()
        >>> arr = NDArray([1, 2, 3], device=cpu_device)
    """

def cuda(index: int = 0) -> Device:
    """
    Create CUDA device.
    
    Args:
        index: GPU device index
        
    Returns:
        CUDA device instance or None if CUDA unavailable
        
    Example:
        >>> gpu_device = genesis.cuda(0)  # First GPU
        >>> if gpu_device.enabled():
        ...     arr = NDArray([1, 2, 3], device=gpu_device)
    """

def device(device_name: Union[str, int]) -> Device:
    """
    Create device by name or index.
    
    Args:
        device_name: 'cpu', 'cuda', 'cuda:N', or GPU index
        
    Returns:
        Device instance
        
    Example:
        >>> dev1 = genesis.device('cuda:1')  # Second GPU
        >>> dev2 = genesis.device(1)         # Same as above
        >>> dev3 = genesis.device('cpu')     # CPU device
    """

def default_device() -> Device:
    """
    Get default device (CPU).
    
    Returns:
        Default device instance
    """

def all_devices() -> List[Device]:
    """
    Get list of all available devices.
    
    Returns:
        List of device instances
    """
```

## Operations

The ndarray system supports a comprehensive set of operations through backend modules.

### Arithmetic Operations

```python
# Binary operations
add(x, y)           # Element-wise addition
sub(x, y)           # Element-wise subtraction  
mul(x, y)           # Element-wise multiplication
truediv(x, y)       # Element-wise division
pow(x, scalar)      # Element-wise power

# Unary operations
log(x)              # Natural logarithm
exp(x)              # Exponential
sin(x)              # Sine
cos(x)              # Cosine
sqrt(x)             # Square root
```

### Reduction Operations

```python
reduce_sum(x, axis=None, keepdims=False)    # Sum reduction
reduce_max(x, axis=None, keepdims=False)    # Max reduction
reduce_min(x, axis=None, keepdims=False)    # Min reduction
```

### Comparison Operations

```python
maximum(x, y)       # Element-wise maximum
minimum(x, y)       # Element-wise minimum
```

### Matrix Operations

```python
matmul(x, y)        # Matrix multiplication
transpose(x, axes)  # Tensor transpose
```

## Performance Optimizations

### Kernel Caching

Triton kernels are automatically cached for reuse:

```python
from genesis.ndarray.kernel_cache import cached_kernel_call

# Kernels are cached by function signature and parameters
cached_kernel_call(kernel_func, grid_func, *args, **kwargs)
```

### Adaptive Configuration

Block sizes automatically adapt to tensor dimensions:

```python
from genesis.ndarray.adaptive_config import AdaptiveConfig

# Get optimized configuration for tensor shape
config = AdaptiveConfig.get_elementwise_config(shape)
block_size = config['BLOCK_SIZE']
grid = config['grid']
```

### Memory Management

Efficient memory allocation patterns:
- Contiguous memory layout when possible
- View-based operations to avoid copying
- Automatic memory cleanup

## GPU Backend (CUDA/Triton)

### GPU Operations

Pure CUDA/Triton implementation for GPU operations optimized for performance.

### Triton Kernels

Hand-optimized Triton kernels for maximum performance:
- Elementwise operations with broadcasting
- Reduction operations with work-efficient algorithms
- Matrix multiplication with tiling
- Memory-optimized access patterns

## Usage Examples

### Basic Array Creation

```python
import genesis

# Create arrays on different devices
cpu_arr = genesis.NDArray([1, 2, 3, 4], device=genesis.cpu())
gpu_arr = genesis.NDArray([1, 2, 3, 4], device=genesis.cuda())

# Create with specific shapes
zeros = genesis.NDArray.make((100, 100), device=genesis.cuda())
zeros.fill(0.0)
```

### Device Operations

```python
# Random arrays
device = genesis.cuda(0)
random_normal = device.randn(1000, 1000)
random_uniform = device.rand(1000, 1000)

# One-hot encoding
indices = genesis.NDArray([0, 2, 1, 3], device=device)
one_hot = device.one_hot(4, indices)
```

### Memory Transfer

```python
# GPU to CPU transfer
gpu_data = genesis.NDArray([1, 2, 3], device=genesis.cuda())
cpu_data = gpu_data.cpu()
numpy_data = gpu_data.numpy()

# CPU to GPU transfer  
cpu_data = genesis.NDArray([1, 2, 3], device=genesis.cpu())
gpu_data = genesis.NDArray(cpu_data, device=genesis.cuda())
```

### Performance Monitoring

```python
import time

# Time operations
start = time.time()
result = genesis.ndarray.add(x, y)
end = time.time()
print(f"Operation took {(end - start) * 1000:.2f}ms")
```

## Backend Selection

Genesis automatically selects the appropriate backend:

```python
# CPU operations use PyTorch backend
cpu_device = genesis.cpu()
x = genesis.NDArray([1, 2, 3], device=cpu_device)

# GPU operations use Triton/CUDA backend
gpu_device = genesis.cuda()
if gpu_device.enabled():
    x = genesis.NDArray([1, 2, 3], device=gpu_device)
    # Uses optimized Triton kernels
```

## Error Handling

The ndarray system provides comprehensive error handling:

```python
try:
    # Attempt GPU operation
    gpu_device = genesis.cuda()
    if not gpu_device.enabled():
        raise RuntimeError("CUDA not available")
    
    arr = genesis.NDArray([1, 2, 3], device=gpu_device)
except RuntimeError as e:
    # Fall back to CPU
    print(f"GPU error: {e}, using CPU")
    cpu_device = genesis.cpu()
    arr = genesis.NDArray([1, 2, 3], device=cpu_device)
```

## Best Practices

1. **Device Selection**: Check device availability before use
2. **Memory Management**: Transfer between devices judiciously  
3. **Batch Operations**: Process multiple tensors together when possible
4. **Contiguous Memory**: Ensure arrays are contiguous for optimal performance
5. **Error Handling**: Always handle CUDA availability gracefully

## Performance Tips

1. **Use appropriate block sizes** for GPU kernels
2. **Minimize device transfers** between CPU and GPU
3. **Leverage kernel caching** by reusing similar operations
4. **Use views instead of copies** when possible
5. **Batch similar operations** to amortize kernel launch overhead

## See Also

- [Tensor Operations](../autograd.md) - High-level tensor interface
- [Neural Network Modules](../nn/modules.md) - Building on ndarray
- [Performance Guide](../../performance/) - Optimization techniques
- [CUDA Storage](../../core-components/cuda-storage.md) - Low-level CUDA implementation