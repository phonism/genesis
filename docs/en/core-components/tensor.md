# Tensor Operations

Core tensor operations and data structures in Genesis.

## Overview

Genesis tensors provide the fundamental data structure for all computations, similar to PyTorch tensors but optimized for our dual backend architecture.

## Creating Tensors

```python
import genesis

# Create tensors
x = genesis.tensor([1, 2, 3, 4])
y = genesis.zeros(3, 4)
z = genesis.randn(2, 3, device='cuda')
```

## Tensor Operations

```python
# Basic operations
result = x + y
result = genesis.matmul(x, y)
result = x.sum(dim=1)
```

## Device Management

```python
# Move tensors between devices
cpu_tensor = gpu_tensor.cpu()
gpu_tensor = cpu_tensor.cuda()
gpu_tensor = cpu_tensor.to('cuda')
```

*This documentation is under construction. More detailed tensor API documentation will be added.*

## See Also

- [Function System](function.md) - Automatic differentiation functions
- [Storage Layer](storage.md) - Memory storage interface
- [Device Management](device.md) - Device abstraction