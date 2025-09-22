# Breaking Changes in Genesis v2.0

This document provides a comprehensive list of all breaking changes introduced in Genesis v2.0.

## 🗑️ Removed Modules and Classes

### NDArray Module (Complete Removal)
**Impact**: High - Affects all code using NDArray directly

```python
# ❌ Removed in v2.0
from genesis.ndarray import NDArray, Device
from genesis.ndarray.cuda_storage import CUDAStorage
from genesis.ndarray.cpu_storage import CPUStorage
from genesis.ndarray.cuda_backend import CUDABackend

# ✅ v2.0 Replacements
import genesis
from genesis.device import Device
from genesis.backends.cuda import CUDAStorage
from genesis.backends.cpu import CPUStorage
```

### Autograd Module Restructure
**Impact**: Medium - Affects custom autograd implementations

```python
# ❌ Removed paths
from genesis.autograd import Variable, Context, Function
from genesis.autograd.engine import backward_engine

# ✅ New paths
from genesis.tensor import Tensor  # Variable -> Tensor
from genesis.function import Function, Context
# backward_engine is now internal
```

### Backend Module Changes
**Impact**: Low - Mainly affects advanced users

```python
# ❌ Old backend system
from genesis.backend import Backend, get_backend
from genesis.backend.registry import register_backend

# ✅ New backend system (mostly internal)
# Backends are automatically managed
# Custom backends: see backend development guide
```

## 🔄 API Changes

### Tensor Creation
**Impact**: Medium - Common operations

```python
# ❌ Old ways that no longer work
x = NDArray([1, 2, 3])
y = Variable([1, 2, 3], requires_grad=True)

# ✅ New unified API
x = genesis.tensor([1, 2, 3])
y = genesis.tensor([1, 2, 3], requires_grad=True)
```

### Device Specification
**Impact**: High - Affects all device-specific code

```python
# ❌ Old device handling
from genesis.ndarray.device import CUDADevice, CPUDevice
device = CUDADevice(0)
tensor = NDArray([1, 2, 3], device=device)

# ✅ New device system
device = genesis.device('cuda:0')
tensor = genesis.tensor([1, 2, 3], device=device)
```

### Operation Interface
**Impact**: Medium - Affects functional programming style

```python
# ❌ Old functional interface
from genesis.functional import add, matmul
result = add(a, b)

# ✅ New interface (both styles supported)
result = genesis.add(a, b)        # Functional style
result = a + b                    # Operator style
```

## 📦 Import Path Changes

### Core Components
```python
# ❌ Old imports
from genesis.autograd import Tensor
from genesis.ndarray import Device
from genesis.backend import get_current_backend

# ✅ New imports
from genesis import Tensor, tensor  # Both available
from genesis.device import Device
# Backend selection is automatic
```

### Neural Network Components
```python
# ❌ Old imports (still work but deprecated)
from genesis.nn.modules.linear import Linear
from genesis.nn.functional import relu

# ✅ Preferred new imports
from genesis.nn import Linear
import genesis.nn.functional as F
```

### Optimization Components
```python
# ✅ These remain the same (no breaking changes)
from genesis.optim import Adam, SGD
from genesis.optim.lr_scheduler import StepLR
```

## 🏗️ Architectural Changes

### Memory Management
**Impact**: Low - Mostly improved, but some advanced usage affected

```python
# ❌ Old memory management
NDArray.set_memory_pool_size(1024 * 1024 * 1024)
NDArray.clear_memory_pool()

# ✅ New memory management
genesis.cuda.set_memory_fraction(0.8)
genesis.cuda.empty_cache()
```

### Context Management
**Impact**: Medium - Affects gradient computation customization

```python
# ❌ Old context usage
class CustomFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input_shape = input.shape
        ctx.save_for_backward(input)
        return input * 2

# ✅ New context usage (similar but enhanced)
class CustomFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_variable('input_shape', input.shape)
        ctx.save_for_backward(input)
        return input * 2
```

## 🔧 Configuration Changes

### Environment Variables
```python
# ❌ Old environment variables (no longer used)
GENESIS_NDARRAY_BACKEND=cuda
GENESIS_DEFAULT_DTYPE=float32

# ✅ New environment variables
GENESIS_DEFAULT_DEVICE=cuda:0
GENESIS_CUDA_MEMORY_FRACTION=0.8
GENESIS_KERNEL_CACHE_DIR=/tmp/genesis_cache
```

### Runtime Configuration
```python
# ❌ Old configuration methods
genesis.set_default_backend('cuda')
genesis.ndarray.set_default_device('cuda:0')

# ✅ New configuration methods
genesis.set_default_device('cuda:0')
genesis.cuda.set_device(0)
```

## ⚙️ Behavioral Changes

### Automatic Type Promotion
**Impact**: Low - Generally improved behavior

```python
# v1.x behavior: Sometimes inconsistent type promotion
a = genesis.tensor([1, 2, 3], dtype=genesis.int32)
b = genesis.tensor([1.0, 2.0, 3.0], dtype=genesis.float32)
c = a + b  # Might fail or produce unexpected types

# v2.0 behavior: Consistent PyTorch-like promotion
a = genesis.tensor([1, 2, 3], dtype=genesis.int32)
b = genesis.tensor([1.0, 2.0, 3.0], dtype=genesis.float32)
c = a + b  # Always produces float32, following clear rules
```

### Device Transfer Behavior
**Impact**: Low - More consistent behavior

```python
# v1.x: Inconsistent behavior with mixed devices
cpu_tensor = genesis.tensor([1, 2, 3], device='cpu')
gpu_tensor = genesis.tensor([4, 5, 6], device='cuda')
result = cpu_tensor + gpu_tensor  # Behavior was unclear

# v2.0: Clear promotion rules
cpu_tensor = genesis.tensor([1, 2, 3], device='cpu')
gpu_tensor = genesis.tensor([4, 5, 6], device='cuda')
result = cpu_tensor + gpu_tensor  # Always promotes to GPU, warns user
```

### Gradient Computation
**Impact**: Low - More efficient, same API

```python
# Same API, but improved performance and memory usage
x = genesis.tensor([1, 2, 3], requires_grad=True)
y = x.sum()
y.backward()  # More efficient in v2.0
```

## 🐛 Bug Fixes That May Affect Code

### Memory Layout Consistency
**Impact**: Low - Better consistency

```python
# v1.x: Inconsistent memory layout
x = genesis.tensor([[1, 2], [3, 4]])
y = x.transpose(0, 1)
# Memory layout might vary between backends

# v2.0: Consistent memory layout
x = genesis.tensor([[1, 2], [3, 4]])
y = x.transpose(0, 1)
# Consistent memory layout across all backends
```

### Broadcasting Rules
**Impact**: Low - More consistent with NumPy/PyTorch

```python
# v1.x: Some edge cases in broadcasting
a = genesis.tensor([1, 2, 3])      # shape: (3,)
b = genesis.tensor([[1], [2]])     # shape: (2, 1)
c = a + b  # Might behave inconsistently

# v2.0: Consistent NumPy/PyTorch-like broadcasting
a = genesis.tensor([1, 2, 3])      # shape: (3,)
b = genesis.tensor([[1], [2]])     # shape: (2, 1)
c = a + b  # shape: (2, 3), consistent behavior
```

## 🔄 Migration Timeline

### Phase 1: Immediate (Required)
- Remove all NDArray imports
- Update tensor creation calls
- Fix device specification

### Phase 2: Short-term (Recommended)
- Update import paths to new preferred locations
- Adopt new configuration methods
- Update custom operations

### Phase 3: Long-term (Optional)
- Leverage new performance features
- Adopt new memory management patterns
- Update testing code

## ⚠️ Common Pitfalls

### Pitfall 1: Assuming Old Import Paths Work
```python
# ❌ This will fail
from genesis.ndarray import NDArray

# ✅ Use this instead
import genesis
x = genesis.tensor(data)
```

### Pitfall 2: Direct Backend Access
```python
# ❌ This pattern no longer works
backend = genesis.get_backend('cuda')
result = backend.add(a, b)

# ✅ Use operation dispatch
result = genesis.add(a, b)  # Automatically uses correct backend
```

### Pitfall 3: Old Memory Management
```python
# ❌ Old memory management doesn't exist
NDArray.set_memory_pool_size(size)

# ✅ Use new memory management
genesis.cuda.set_memory_fraction(0.8)
```

## 🛠️ Migration Tools

### Automated Migration Script
We provide a migration script to help with common patterns:

```bash
# Run the migration script on your codebase
python -m genesis.migrate /path/to/your/code

# Preview changes without applying
python -m genesis.migrate /path/to/your/code --dry-run

# Migrate specific patterns only
python -m genesis.migrate /path/to/your/code --only=imports,tensor_creation
```

### Manual Migration Checklist
- [ ] Remove NDArray imports
- [ ] Update tensor creation
- [ ] Fix device specifications
- [ ] Update import paths
- [ ] Test functionality
- [ ] Update configuration
- [ ] Verify performance

## 📞 Support

If you encounter issues with breaking changes:

1. **Check this document first** - It covers most common issues
2. **Use the migration script** - Automates many common changes
3. **Check the migration guide** - Provides detailed examples
4. **File an issue** - If you find undocumented breaking changes

## 🔗 Related Documentation

- [Migration Guide](v2-migration.md) - Step-by-step migration process
- [Backend System](../backends/index.md) - Understanding the new architecture
- [Device Management](../core-components/device.md) - New device system
- [API Reference](../api-reference/index.md) - Complete v2.0 API documentation