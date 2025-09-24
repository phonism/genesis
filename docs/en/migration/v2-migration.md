# Migration Guide: Genesis v1 to v2

This guide helps you migrate your code from Genesis v1.x to Genesis v2.0, which introduces significant architectural changes.

## 📋 Overview

Genesis v2.0 introduces a major architectural overhaul with:
- **Modular Backend System**: Separated CPU and CUDA implementations
- **Unified Device Abstraction**: Consistent device management
- **Operation Dispatch System**: Centralized operation routing
- **Removed NDArray Module**: Functionality moved to backends

### 🆕 v2.0.1 Code Quality Enhancements
The latest release focuses on production-ready code quality:
- **Complete Documentation**: 100% docstring coverage for all public APIs
- **Type Safety**: Comprehensive type annotations throughout the codebase
- **Cleaner Architecture**: Refactored complex functions for better maintainability
- **Zero Function-Level Imports**: Eliminated problematic import patterns
- **Enhanced Error Handling**: Robust validation and graceful error recovery

## 🔄 Breaking Changes Summary

### 1. NDArray Module Removed
The entire `genesis.ndarray` module has been removed and its functionality integrated into the new backend system.

#### Before (v1.x)
```python
# NDArray was explicitly used
from genesis.ndarray import NDArray
x = NDArray([1, 2, 3], device='cuda')
```

#### After (v2.0)
```python
# Direct tensor creation
import genesis
x = genesis.tensor([1, 2, 3], device='cuda')
```

### 2. Import Changes
Many import paths have changed due to restructuring.

#### Before (v1.x)
```python
from genesis.autograd import Tensor
from genesis.ndarray.device import Device
from genesis.ndarray.cuda_storage import CUDAStorage
```

#### After (v2.0)
```python
from genesis import Tensor  # or just use genesis.tensor()
from genesis.device import Device
from genesis.backends.cuda import CUDAStorage
```

### 3. Backend Access
Direct backend access has been restructured.

#### Before (v1.x)
```python
# Accessing CUDA functions directly
from genesis.ndarray.cuda_backend import cuda_add
result = cuda_add(a, b)
```

#### After (v2.0)
```python
# Operations automatically dispatch to correct backend
result = genesis.add(a, b)  # Automatically uses CUDA if tensors are on GPU
```

## 🔧 Code Migration Steps

### Step 1: Update Imports
Replace old imports with new ones:

```python
# Old imports (v1.x) -> New imports (v2.0)
from genesis.autograd import Tensor          -> from genesis import tensor, Tensor
from genesis.ndarray import NDArray          -> # Remove, use genesis.tensor()
from genesis.ndarray.device import Device    -> from genesis.device import Device
from genesis.backend import Backend          -> # Remove, handled automatically
```

### Step 2: Replace NDArray Usage
Convert NDArray operations to tensor operations:

```python
# Before (v1.x)
def old_function():
    x = NDArray([1, 2, 3], device='cuda')
    y = NDArray([4, 5, 6], device='cuda')
    return x.add(y)

# After (v2.0)
def new_function():
    x = genesis.tensor([1, 2, 3], device='cuda')
    y = genesis.tensor([4, 5, 6], device='cuda')
    return x + y  # or genesis.add(x, y)
```

### Step 3: Update Device Handling
Use the new unified device system:

```python
# Before (v1.x)
from genesis.ndarray.device import get_device
device = get_device('cuda:0')

# After (v2.0)
device = genesis.device('cuda:0')
```

### Step 4: Backend-Specific Code
If you had backend-specific code, update it:

```python
# Before (v1.x) - Direct backend access
from genesis.ndarray.cuda_backend import CUDABackend
backend = CUDABackend()
result = backend.matmul(a, b)

# After (v2.0) - Use operation dispatch
result = genesis.matmul(a, b)  # Automatically routed to appropriate backend
```

## 📝 Common Migration Patterns

### Pattern 1: Tensor Creation
```python
# Before (v1.x)
def create_tensors_v1():
    x = NDArray([1, 2, 3])
    y = NDArray.zeros((3, 3))
    z = NDArray.ones((2, 2), device='cuda')
    return x, y, z

# After (v2.0)
def create_tensors_v2():
    x = genesis.tensor([1, 2, 3])
    y = genesis.zeros((3, 3))
    z = genesis.ones((2, 2), device='cuda')
    return x, y, z
```

### Pattern 2: Device Transfer
```python
# Before (v1.x)
def transfer_v1(tensor):
    cuda_tensor = tensor.to_device('cuda')
    cpu_tensor = cuda_tensor.to_device('cpu')
    return cpu_tensor

# After (v2.0)
def transfer_v2(tensor):
    cuda_tensor = tensor.to('cuda')
    cpu_tensor = cuda_tensor.to('cpu')
    return cpu_tensor
```

### Pattern 3: Custom Operations
```python
# Before (v1.x) - Required NDArray knowledge
def custom_op_v1(x):
    if x.device.is_cuda:
        result = cuda_custom_kernel(x.data)
    else:
        result = cpu_custom_impl(x.data)
    return NDArray(result, device=x.device)

# After (v2.0) - Use operation dispatch
def custom_op_v2(x):
    return genesis.ops.custom_operation(x)  # Automatically dispatched
```

### Pattern 4: Memory Management
```python
# Before (v1.x)
def manage_memory_v1():
    x = NDArray.zeros((1000, 1000), device='cuda')
    # Manual memory management
    del x
    NDArray.cuda_empty_cache()

# After (v2.0)
def manage_memory_v2():
    x = genesis.zeros((1000, 1000), device='cuda')
    # Improved automatic memory management
    del x
    genesis.cuda.empty_cache()  # Still available but less needed
```

## ⚠️ Potential Issues and Solutions

### Issue 1: Import Errors
**Problem**: `ImportError: cannot import name 'NDArray'`

**Solution**: Replace NDArray usage with tensor functions
```python
# Fix import error
# from genesis.ndarray import NDArray  # Remove this line
import genesis
x = genesis.tensor(data)  # Use this instead
```

### Issue 2: Device Attribute Errors
**Problem**: `AttributeError: 'str' object has no attribute 'is_cuda'`

**Solution**: Use proper device objects
```python
# Before - device was sometimes a string
device = 'cuda'
if device == 'cuda':  # String comparison

# After - use device objects
device = genesis.device('cuda')
if device.is_cuda:  # Proper attribute
```

### Issue 3: Backend Method Not Found
**Problem**: Direct backend method calls fail

**Solution**: Use the operation dispatch system
```python
# Before - direct backend call
result = backend.specific_operation(x)

# After - use dispatched operation
result = genesis.ops.specific_operation(x)
```

### Issue 4: Performance Regression
**Problem**: Code runs slower after migration

**Solution**:
1. Ensure tensors are on the correct device
2. Use in-place operations where possible
3. Check for unnecessary device transfers

```python
# Check tensor device placement
print(f"Tensor device: {x.device}")

# Use in-place operations
x.add_(y)  # Instead of x = x + y

# Minimize device transfers
# Keep related tensors on the same device
```

## ✅ Migration Checklist

Use this checklist to ensure complete migration:

- [ ] **Remove NDArray imports**
  - [ ] Remove `from genesis.ndarray import NDArray`
  - [ ] Remove `from genesis.ndarray.device import Device`
  - [ ] Remove other ndarray-specific imports

- [ ] **Update tensor creation**
  - [ ] Replace `NDArray(data)` with `genesis.tensor(data)`
  - [ ] Replace `NDArray.zeros()` with `genesis.zeros()`
  - [ ] Replace `NDArray.ones()` with `genesis.ones()`

- [ ] **Update device handling**
  - [ ] Use `genesis.device()` for device creation
  - [ ] Update device attribute access
  - [ ] Check device transfer methods

- [ ] **Update operations**
  - [ ] Replace direct backend calls with operation dispatch
  - [ ] Update custom operation implementations
  - [ ] Verify operation behavior consistency

- [ ] **Test functionality**
  - [ ] Run existing tests
  - [ ] Verify performance characteristics
  - [ ] Check memory usage patterns

## 🚀 Taking Advantage of New Features

### Enhanced Memory Management
```python
# Take advantage of improved memory pooling
genesis.cuda.set_memory_fraction(0.9)  # Use 90% of GPU memory

# Monitor memory usage
stats = genesis.cuda.memory_stats()
print(f"Memory efficiency: {stats.fragmentation_ratio:.2%}")
```

### Improved Device Management
```python
# Use automatic device selection
device = genesis.device('auto')  # Chooses best available device

# Context-based device management
with genesis.device('cuda:1'):
    x = genesis.randn(1000, 1000)  # Automatically on cuda:1
```

### Operation Profiling
```python
# Profile operations
with genesis.ops.profile() as prof:
    result = complex_computation(data)

prof.print_stats()  # See performance breakdown
```

## 🔗 Additional Resources

- [Breaking Changes](breaking-changes.md) - Complete list of breaking changes
- [Backend System](../backends/index.md) - Understanding the new backend architecture
- [Device Guide](../core-components/device.md) - Device management in v2.0
- [Performance Guide](../performance/optimization-guide.md) - Optimizing v2.0 code

## 💡 Getting Help

If you encounter issues during migration:

1. **Check the documentation** - Most common patterns are covered
2. **Search issues** - Look for similar problems in GitHub issues
3. **Ask questions** - Create a new issue with the "migration" label
4. **Provide examples** - Include before/after code snippets

Remember that v2.0 provides better performance and cleaner architecture, so the migration effort is worthwhile!