# CUDA Storage System

Genesis's CUDA Storage (CUDAStorage) is a core component of the framework, providing pure CUDA implementation for GPU memory management and operations, completely independent of PyTorch, using CUDA Python API directly.

## 🎯 Design Goals

### Independence
- **Pure CUDA Implementation**: No dependency on PyTorch's GPU backend
- **Direct Memory Management**: Direct GPU memory management using CUDA Python API
- **High Performance**: Memory access patterns optimized for GPU

### Compatibility  
- **PyTorch-style API**: Maintains interface compatibility with PyTorch tensors
- **Automatic Differentiation Support**: Seamless integration with Genesis's autograd system
- **Type Safety**: Complete type annotations and runtime checking

## 🏗️ Architecture Design

### IndexPlan Architecture

CUDATensor uses an advanced IndexPlan architecture to handle complex tensor indexing operations:

```python
class IndexKind(Enum):
    VIEW = "view"           # Pure view operation, zero-copy
    GATHER = "gather"       # Gather operation for advanced indexing  
    SCATTER = "scatter"     # Scatter operation for assignment
    COPY = "copy"          # Strided copy
    FILL = "fill"          # Fill operation

@dataclass
class IndexPlan:
    """Unified index plan"""
    kind: IndexKind
    result_shape: Optional[Tuple[int, ...]] = None
    result_strides: Optional[Tuple[int, ...]] = None
    ptr_offset_bytes: int = 0
    index_tensor: Optional['CUDATensor'] = None
    needs_mask_compaction: bool = False
    temp_memory_bytes: int = 0
```

### Memory Management

```python
class AsyncMemoryPool:
    """Asynchronous memory pool to optimize GPU memory allocation performance"""
    
    def __init__(self):
        self.free_blocks = {}  # Free blocks organized by size
        self.allocated_blocks = {}  # Allocated blocks
        self.alignment = 512  # Memory alignment, consistent with PyTorch
        
    def allocate(self, size_bytes: int) -> int:
        """Allocate aligned GPU memory"""
        
    def deallocate(self, ptr: int):
        """Release GPU memory to pool for reuse"""
```

## 💡 Core Features

### 1. Efficient Indexing Operations

```python
import genesis

# Create CUDA tensor
x = genesis.randn(1000, 1000, device='cuda')

# Basic indexing - uses VIEW operation, zero-copy
y = x[10:20, 50:100]  # IndexPlan.kind = VIEW

# Advanced indexing - uses GATHER operation  
indices = genesis.tensor([1, 3, 5, 7], device='cuda')
z = x[indices]  # IndexPlan.kind = GATHER

# Boolean indexing - automatic optimization
mask = x > 0.5
w = x[mask]  # Choose optimal strategy based on sparsity
```

### 2. Memory-Efficient Operations

```python
# In-place operations, avoid memory allocation
x = genesis.randn(1000, 1000, device='cuda')
x += 1.0  # In-place addition

# View operations, zero-copy
y = x.view(100, 10000)  # Change shape without copying data
z = x.transpose(0, 1)   # Transpose view

# Strided operations, efficient implementation
w = x[::2, ::3]  # Strided indexing using optimized COPY operation
```

### 3. Triton Kernel Integration

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized Triton addition kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

# CUDATensor automatically calls optimized Triton kernel
def add_cuda_tensor(x: CUDATensor, y: CUDATensor) -> CUDATensor:
    """CUDA tensor addition using Triton optimization"""
    output = CUDATensor(x.shape, x.dtype)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](x.data_ptr(), y.data_ptr(), output.data_ptr(), 
                     n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
```

## 🚀 Basic Usage

### Creating Tensors

```python
import genesis

# Create from data
data = [[1.0, 2.0], [3.0, 4.0]]
tensor = genesis.tensor(data, device='cuda')

# Create specific shapes directly
zeros = genesis.zeros(100, 100, device='cuda')
ones = genesis.ones(50, 50, device='cuda')  
random = genesis.randn(200, 200, device='cuda')

# Specify data type
float16_tensor = genesis.randn(100, 100, dtype=genesis.float16, device='cuda')
int_tensor = genesis.randint(0, 10, (50, 50), device='cuda')

print(f"Tensor shape: {tensor.shape}")
print(f"Data type: {tensor.dtype}")
print(f"Device: {tensor.device}")
print(f"Strides: {tensor.strides}")
```

### Basic Operations

```python
# Mathematical operations
x = genesis.randn(100, 100, device='cuda')
y = genesis.randn(100, 100, device='cuda')

# Element-wise operations
z = x + y      # Addition
w = x * y      # Multiplication  
u = x.pow(2)   # Power operation
v = x.exp()    # Exponential function

# Reduction operations
sum_all = x.sum()           # Global sum
sum_dim = x.sum(dim=0)      # Sum along dimension
mean_val = x.mean()         # Mean value
max_val, indices = x.max(dim=1)  # Maximum value and indices

# Linear algebra
a = genesis.randn(100, 50, device='cuda')
b = genesis.randn(50, 200, device='cuda') 
c = genesis.matmul(a, b)    # Matrix multiplication

# Shape operations
reshaped = x.view(10, 1000)        # Reshape
transposed = x.transpose(0, 1)     # Transpose  
flattened = x.flatten()            # Flatten
```

### Advanced Indexing

```python
# Create test tensor
data = genesis.arange(0, 100, device='cuda').view(10, 10)
print("Original data:")
print(data)

# Basic slicing
slice_basic = data[2:5, 3:7]  # Rows 2-4, columns 3-6
print("Basic slicing:", slice_basic.shape)

# Strided indexing
slice_stride = data[::2, 1::2]  # Every other row, every other column starting from column 1
print("Strided indexing:", slice_stride.shape)

# Advanced indexing - integer arrays
row_indices = genesis.tensor([0, 2, 4, 6], device='cuda')
col_indices = genesis.tensor([1, 3, 5, 7], device='cuda')
advanced = data[row_indices, col_indices]  # Select specific positions
print("Advanced indexing result:", advanced)

# Boolean indexing
mask = data > 50
masked_data = data[mask]  # Select elements greater than 50
print("Boolean indexing result:", masked_data)

# Mixed indexing
mixed = data[row_indices, 2:8]  # Column range for specific rows
print("Mixed indexing:", mixed.shape)
```

## 🔧 Memory Management

### Memory Pool Optimization

```python
# Check memory usage
print(f"Allocated memory: {genesis.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Cached memory: {genesis.cuda.memory_cached() / 1024**2:.1f} MB")

# Manual memory management
x = genesis.randn(1000, 1000, device='cuda')
print(f"After tensor creation: {genesis.cuda.memory_allocated() / 1024**2:.1f} MB")

del x  # Delete reference
genesis.cuda.empty_cache()  # Empty cache
print(f"After cleanup: {genesis.cuda.memory_allocated() / 1024**2:.1f} MB")

# Memory snapshot (for debugging)
snapshot = genesis.cuda.memory_snapshot()
for entry in snapshot[:3]:  # Show first 3 entries
    print(f"Address: {entry['address']}, Size: {entry['size']} bytes")
```

### Asynchronous Operations

```python
# Asynchronous memory operations
with genesis.cuda.stream():
    x = genesis.randn(1000, 1000, device='cuda')
    y = genesis.randn(1000, 1000, device='cuda')
    z = genesis.matmul(x, y)  # Asynchronous execution
    
    # Other CPU work can proceed in parallel
    print("Matrix multiplication running asynchronously on GPU...")
    
    # Synchronize and wait for results  
    genesis.cuda.synchronize()
    print("Computation complete:", z.shape)
```

## ⚡ Performance Optimization

### 1. Memory Access Pattern Optimization

```python
def inefficient_access():
    """Inefficient memory access pattern"""
    x = genesis.randn(1000, 1000, device='cuda')
    result = genesis.zeros(1000, device='cuda')
    
    # Non-contiguous access, cache misses
    for i in range(1000):
        result[i] = x[i, ::10].sum()  # Strided access
    
    return result

def efficient_access():  
    """Efficient memory access pattern"""
    x = genesis.randn(1000, 1000, device='cuda')
    
    # Contiguous access, full cache utilization
    indices = genesis.arange(0, 1000, 10, device='cuda')
    selected = x[:, indices]  # Batch selection
    result = selected.sum(dim=1)  # Vectorized summation
    
    return result

# Performance comparison
import time

start = time.time()
result1 = inefficient_access()
time1 = time.time() - start

start = time.time()  
result2 = efficient_access()
time2 = time.time() - start

print(f"Inefficient method: {time1:.4f}s")
print(f"Efficient method: {time2:.4f}s")  
print(f"Speedup: {time1/time2:.2f}x")
```

### 2. Batch Operations Optimization

```python
def batch_operations_demo():
    """Demonstrate performance advantages of batch operations"""
    
    # Create test data
    matrices = [genesis.randn(100, 100, device='cuda') for _ in range(10)]
    
    # Method 1: Individual processing (inefficient)
    start = time.time()
    results1 = []
    for matrix in matrices:
        result = matrix.exp().sum()
        results1.append(result)
    time1 = time.time() - start
    
    # Method 2: Batch processing (efficient)
    start = time.time()
    batched = genesis.stack(matrices, dim=0)  # [10, 100, 100]
    results2 = batched.exp().sum(dim=(1, 2))  # [10]
    time2 = time.time() - start
    
    print(f"Individual processing: {time1:.4f}s")
    print(f"Batch processing: {time2:.4f}s")
    print(f"Speedup: {time1/time2:.2f}x")

batch_operations_demo()
```

### 3. In-place Operations

```python
def inplace_operations_demo():
    """Demonstrate memory efficiency of in-place operations"""
    
    # Non-in-place operations (create new tensors)
    x = genesis.randn(1000, 1000, device='cuda')
    start_memory = genesis.cuda.memory_allocated()
    
    y = x + 1.0      # Create new tensor
    z = y * 2.0      # Create another new tensor
    w = z.exp()      # Create yet another new tensor
    
    memory_after = genesis.cuda.memory_allocated()
    print(f"Non-in-place operations memory growth: {(memory_after - start_memory) / 1024**2:.1f} MB")
    
    # In-place operations (modify original tensor)
    x = genesis.randn(1000, 1000, device='cuda')
    start_memory = genesis.cuda.memory_allocated()
    
    x += 1.0         # In-place addition
    x *= 2.0         # In-place multiplication  
    x.exp_()         # In-place exponential function
    
    memory_after = genesis.cuda.memory_allocated()
    print(f"In-place operations memory growth: {(memory_after - start_memory) / 1024**2:.1f} MB")

inplace_operations_demo()
```

## 🐛 Debugging and Diagnostics

### Memory Leak Detection

```python
def detect_memory_leaks():
    """Detect memory leaks"""
    genesis.cuda.empty_cache()
    initial_memory = genesis.cuda.memory_allocated()
    
    # Perform some operations
    for i in range(100):
        x = genesis.randn(100, 100, device='cuda')
        y = x.matmul(x)
        del x, y
    
    genesis.cuda.empty_cache()
    final_memory = genesis.cuda.memory_allocated()
    
    if final_memory > initial_memory:
        print(f"Possible memory leak: {(final_memory - initial_memory) / 1024**2:.1f} MB")
    else:
        print("No memory leak detected")

detect_memory_leaks()
```

### Error Diagnostics

```python
def diagnose_cuda_errors():
    """CUDA error diagnostics"""
    try:
        # Operations that might cause errors
        x = genesis.randn(1000, 1000, device='cuda')
        y = genesis.randn(500, 500, device='cuda')  # Shape mismatch
        z = genesis.matmul(x, y)
        
    except RuntimeError as e:
        print(f"CUDA error: {e}")
        
        # Check CUDA status
        if genesis.cuda.is_available():
            print(f"CUDA device: {genesis.cuda.get_device_name()}")
            print(f"CUDA capability: {genesis.cuda.get_device_capability()}")
            print(f"Available memory: {genesis.cuda.get_device_properties().total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA unavailable")

diagnose_cuda_errors()
```

## 🔄 PyTorch Interoperability

```python
import torch

def pytorch_interop_demo():
    """Demonstrate interoperability with PyTorch"""
    
    # Convert Genesis tensor to PyTorch
    genesis_tensor = genesis.randn(100, 100, device='cuda')
    
    # Convert to PyTorch (shared memory)
    pytorch_tensor = torch.as_tensor(genesis_tensor.detach().cpu().numpy()).cuda()
    
    print(f"Genesis shape: {genesis_tensor.shape}")
    print(f"PyTorch shape: {pytorch_tensor.shape}")
    
    # PyTorch tensor to Genesis  
    torch_data = torch.randn(50, 50, device='cuda')
    genesis_from_torch = genesis.tensor(torch_data.cpu().numpy(), device='cuda')
    
    print(f"Conversion successful, Genesis tensor: {genesis_from_torch.shape}")

pytorch_interop_demo()
```

## 📊 Performance Benchmarks

```python
def benchmark_cuda_tensor():
    """CUDA tensor performance benchmark tests"""
    
    sizes = [100, 500, 1000, 2000]
    
    print("Matrix multiplication performance comparison (Genesis vs PyTorch):")
    print("-" * 50)
    
    for size in sizes:
        # Genesis test
        x_gen = genesis.randn(size, size, device='cuda')
        y_gen = genesis.randn(size, size, device='cuda')
        
        genesis.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z_gen = genesis.matmul(x_gen, y_gen)
        genesis.cuda.synchronize()
        genesis_time = (time.time() - start) / 10
        
        # PyTorch test
        x_torch = torch.randn(size, size, device='cuda')
        y_torch = torch.randn(size, size, device='cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z_torch = torch.matmul(x_torch, y_torch)
        torch.cuda.synchronize() 
        pytorch_time = (time.time() - start) / 10
        
        ratio = genesis_time / pytorch_time
        print(f"{size}x{size}: Genesis {genesis_time:.4f}s, PyTorch {pytorch_time:.4f}s, ratio {ratio:.2f}")

benchmark_cuda_tensor()
```

## 🎯 Best Practices

### 1. Memory Management Best Practices

```python
# ✅ Good practices
def good_memory_practice():
    with genesis.cuda.device(0):  # Explicitly specify device
        x = genesis.randn(1000, 1000, device='cuda')
        
        # Use in-place operations
        x += 1.0
        x *= 0.5
        
        # Release large tensors promptly
        del x
        genesis.cuda.empty_cache()

# ❌ Practices to avoid  
def bad_memory_practice():
    tensors = []
    for i in range(100):
        x = genesis.randn(1000, 1000, device='cuda')
        y = x + 1.0  # Create additional copy
        tensors.append(y)  # Keep all references, memory cannot be freed
    # Memory will be exhausted quickly
```

### 2. Performance Optimization Best Practices

```python
# ✅ Vectorized operations
def vectorized_operations():
    x = genesis.randn(1000, 1000, device='cuda')
    
    # Use vectorized functions
    result = genesis.relu(x).sum(dim=1).mean()
    
# ❌ Avoid loops
def avoid_loops():
    x = genesis.randn(1000, 1000, device='cuda')
    
    # Avoid Python loops
    result = 0
    for i in range(1000):
        result += x[i].sum()  # Launches CUDA kernel each time
```

### 3. Debugging Best Practices

```python
# Enable CUDA error checking
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Use assertions to check tensor properties
def safe_tensor_operation(x, y):
    assert x.device == y.device, "Tensors must be on the same device"
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    
    return x + y
```

## ❓ Common Issues

### Q: What to do when CUDA memory is insufficient?
A: 
```python
# Reduce batch size
batch_size = 32  # Change to 16 or 8

# Use gradient accumulation
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

# Enable mixed precision
x = genesis.randn(1000, 1000, dtype=genesis.float16, device='cuda')

# Regularly clean memory
genesis.cuda.empty_cache()
```

### Q: Why are CUDA operations slow?  
A: Check the following points:
```python
# 1. Ensure tensors are on GPU
assert x.device.type == 'cuda'

# 2. Avoid frequent CPU-GPU transfers
# Wrong approach
for i in range(1000):
    cpu_data = x.cpu().numpy()  # Transfer each time

# Correct approach
cpu_data = x.cpu().numpy()  # Transfer only once

# 3. Use appropriate data types
x = genesis.randn(1000, 1000, dtype=genesis.float16, device='cuda')  # Faster
```

### Q: How to debug CUDA kernel errors?
A:
```python
# 1. Enable synchronous error checking
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 2. Check tensor validity
def check_tensor(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")

# 3. Use CUDA debugging tools
# cuda-memcheck python your_script.py
# compute-sanitizer python your_script.py
```

---

!!! tip "Performance Tip"
    CUDA tensor performance largely depends on memory access patterns and the use of batch operations. Prioritize vectorized operations and reasonable memory layout.

**Ready to learn more?**

[Next: Tensor Operations Guide](tensor.md){ .md-button .md-button--primary }
[Back to Core Components](index.md){ .md-button }