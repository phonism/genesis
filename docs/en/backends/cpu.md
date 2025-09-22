# CPU Backend

The CPU backend provides efficient CPU-based tensor operations by leveraging PyTorch's optimized CPU kernels.

## 📋 Overview

The CPU backend (`backends/cpu.py`) serves as:
- The default backend for CPU operations
- A reference implementation for new backends
- A fallback when GPU is unavailable

## 🏗️ Architecture

```python
# backends/cpu.py structure
class CPUStorage:
    """CPU tensor storage implementation."""

    def __init__(self, data):
        """Initialize with PyTorch tensor."""
        self.data = data  # PyTorch tensor

    def to(self, device):
        """Transfer to another device."""
        ...

    def copy_(self, other):
        """In-place copy from another storage."""
        ...
```

## 🎯 Key Features

### PyTorch Integration
- Leverages PyTorch's mature CPU implementations
- Benefits from PyTorch's optimizations (MKL, OpenMP)
- Compatible with PyTorch tensors for interoperability

### Operation Support
The CPU backend supports all fundamental operations:

| Category | Operations |
|----------|-----------|
| **Arithmetic** | add, subtract, multiply, divide, power |
| **Reduction** | sum, mean, max, min, argmax, argmin |
| **Matrix** | matmul, transpose, reshape, flatten |
| **Activation** | relu, sigmoid, tanh, softmax |
| **Comparison** | eq, ne, lt, le, gt, ge |

### Memory Management
- Direct memory access without pooling (handled by PyTorch)
- Efficient memory layout for cache optimization
- Support for various data types (float32, float64, int32, etc.)

## 💻 Implementation Details

### Storage Creation
```python
def create_cpu_storage(data, dtype=None):
    """Create CPU storage from various input types."""
    if isinstance(data, torch.Tensor):
        tensor = data.cpu()
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)

    if dtype:
        tensor = tensor.to(dtype)

    return CPUStorage(tensor)
```

### Operation Dispatch
Operations are dispatched through the unified operation system:
```python
# ops/cpu/basic.py
def cpu_add(a, b):
    """CPU implementation of addition."""
    return a.data + b.data

def cpu_matmul(a, b):
    """CPU implementation of matrix multiplication."""
    return torch.matmul(a.data, b.data)
```

## 🚀 Performance Considerations

### Optimization Strategies
1. **Vectorization**: Leverage SIMD instructions through PyTorch
2. **Parallelization**: Utilize multiple CPU cores via OpenMP
3. **Cache Efficiency**: Optimize memory access patterns

### Performance Tips
- Use contiguous memory layouts for better cache utilization
- Batch operations to reduce overhead
- Consider memory pinning for CPU-GPU transfers

## 🔧 Configuration

### Environment Variables
```bash
# Control CPU thread count
export OMP_NUM_THREADS=8

# Enable MKL optimizations
export MKL_NUM_THREADS=8
```

### Runtime Configuration
```python
import genesis

# Set CPU backend as default
genesis.set_default_device("cpu")

# Create CPU tensors
x = genesis.tensor([1, 2, 3])  # Uses CPU backend
```

## 📊 Benchmarks

Relative performance compared to pure PyTorch:

| Operation | Size | Genesis CPU | PyTorch | Ratio |
|-----------|------|-------------|---------|-------|
| Add | 1M | 1.05x | 1.0x | 1.05 |
| MatMul | 1024x1024 | 0.98x | 1.0x | 0.98 |
| Softmax | 10K | 1.02x | 1.0x | 1.02 |

*Note: Near-identical performance due to PyTorch backend*

## 🔍 Debugging

Enable debug mode for CPU operations:
```python
import genesis
genesis.backends.cpu.debug_mode = True

# Now operations will print debug information
x = genesis.tensor([1, 2, 3], device="cpu")
y = x + 1  # Prints: "CPU Add: shape=(3,), dtype=float32"
```

## 🔗 See Also

- [Backend System Overview](index.md)
- [CUDA Backend](cuda.md)
- [Operation Dispatch](../ops/dispatcher.md)