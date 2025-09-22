# Storage API Reference

The Storage layer provides the low-level memory management interface between tensors and device backends in Genesis.

## Storage Class

The Storage class is the base abstraction for memory storage across different devices.

```python
class Storage:
    """
    Abstract base class for tensor storage.
    Provides unified interface for memory management across devices.
    """
```

### Properties

- **size**: Number of elements in storage
- **dtype**: Data type of stored elements
- **device**: Device where storage resides
- **data_ptr**: Pointer to underlying memory (device-specific)

### Methods

```python
storage.to(device)           # Move storage to different device
storage.copy_(src)           # Copy data from another storage
storage.resize_(size)        # Resize storage capacity
storage.fill_(value)         # Fill storage with value
storage.zero_()             # Zero out storage
```

## Backend Implementations

### CPU Storage

CPU storage uses PyTorch tensors as the underlying storage mechanism:

```python
# Internal implementation (not directly accessible)
class CPUStorage(Storage):
    def __init__(self, data):
        self._data = torch.tensor(data)
```

### CUDA Storage

CUDA storage manages GPU memory directly:

```python
# Internal implementation (not directly accessible)
class CUDAStorage(Storage):
    def __init__(self, size, dtype):
        self._allocate_cuda_memory(size, dtype)
```

## Storage Creation

Storage is typically created automatically when creating tensors:

```python
import genesis

# Storage created automatically
tensor = genesis.tensor([1, 2, 3, 4])

# Access underlying storage (advanced usage)
storage = tensor._storage  # Internal API
```

## Memory Layout

### Contiguous Storage

Genesis ensures tensors are stored contiguously in memory for optimal performance:

```python
# Check if tensor is contiguous
is_contiguous = tensor.is_contiguous()

# Make tensor contiguous if needed
contiguous_tensor = tensor.contiguous()
```

### Strided Storage

Storage supports strided access for efficient view operations:

```python
# Reshape creates a view with different strides
reshaped = tensor.reshape(2, 2)

# Transpose changes strides without copying data
transposed = tensor.T
```

## Memory Pooling

Genesis implements memory pooling for efficient allocation:

### CUDA Memory Pool

```python
import genesis.cuda as cuda

# Memory pool statistics (if available)
if genesis.cuda_available():
    # Get memory stats
    allocated = cuda.memory_allocated()
    reserved = cuda.memory_reserved()
```

## Advanced Usage

### Custom Storage Allocation

For advanced users who need custom memory management:

```python
import genesis
import numpy as np

# Create tensor from existing memory
numpy_array = np.array([1, 2, 3, 4], dtype=np.float32)
tensor = genesis.from_numpy(numpy_array)

# Share memory with numpy (zero-copy)
shared_array = tensor.numpy()  # Shares memory if on CPU
```

### Storage Aliasing

Multiple tensors can share the same storage:

```python
# Create view that shares storage
x = genesis.tensor([[1, 2], [3, 4]])
y = x.view(-1)  # Flattened view

# Both tensors share same storage
x[0, 0] = 10
print(y[0])  # 10 - storage is shared
```

### Storage Cloning

Create independent copy with new storage:

```python
# Clone creates new storage
x = genesis.tensor([1, 2, 3])
y = x.clone()

# Modifications don't affect original
y[0] = 10
print(x[0])  # Still 1 - different storage
```

## Memory Management Best Practices

### 1. Reuse Storage

```python
# Reuse existing tensor storage
output = genesis.empty_like(input)
# Perform operation in-place
output.copy_(process(input))
```

### 2. Avoid Unnecessary Copies

```python
# Use views when possible
batch = data.view(batch_size, -1)  # No copy

# Avoid unnecessary contiguous calls
if not tensor.is_contiguous():
    tensor = tensor.contiguous()
```

### 3. Free Unused Storage

```python
# Explicitly free large tensors
large_tensor = genesis.randn(10000, 10000)
# Use tensor
result = process(large_tensor)
# Free memory
del large_tensor
```

### 4. Monitor Memory Usage

```python
import genesis.cuda as cuda

def memory_summary():
    if genesis.cuda_available():
        print(f"Allocated: {cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved: {cuda.memory_reserved() / 1024**2:.2f} MB")
```

## Storage and Gradients

Storage interacts with automatic differentiation:

```python
# Gradient storage allocated on demand
x = genesis.tensor([1, 2, 3], requires_grad=True)
y = x * 2
y.backward(genesis.ones_like(y))

# Gradient has its own storage
print(x.grad)  # Separate storage for gradients
```

## Platform-Specific Considerations

### CPU Storage
- Uses system RAM
- Supports memory-mapped files for large datasets
- Thread-safe operations

### CUDA Storage
- Limited by GPU memory
- Supports unified memory on compatible hardware
- Asynchronous operations possible

## See Also

- [Tensor API](tensor.md) - High-level tensor interface
- [Device Management](device.md) - Device abstraction
- [Memory Management](memory.md) - Advanced memory optimization
- [Backend System](../backends/index.md) - Backend implementations