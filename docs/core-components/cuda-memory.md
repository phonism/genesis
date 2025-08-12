# CUDA Memory Management

Genesis includes a custom CUDA memory management system that provides reliable GPU memory allocation for all tensor operations.

## Overview

The CUDA Memory Manager handles GPU memory allocation and deallocation through direct CUDA Driver API calls, ensuring stability and correctness over performance optimization.

## Architecture

### Core Components

#### CUDAMemoryManager
The main memory manager class that handles:
- CUDA context initialization using primary context
- Direct memory allocation via `cuMemAlloc`
- Direct memory deallocation via `cuMemFree`
- Basic statistics tracking

#### Key Features

1. **Primary Context Usage**
   - Uses `cuDevicePrimaryCtxRetain` instead of `cuCtxCreate`
   - Avoids API compatibility issues
   - Ensures proper context sharing

2. **Direct Memory Operations**
   - No memory alignment or padding
   - No caching or memory pools
   - Direct `cuMemAlloc`/`cuMemFree` calls
   - Prioritizes correctness over performance

3. **Stream Management**
   - Creates and manages default CUDA stream
   - Proper resource cleanup in destructor

## Usage

The memory manager is used internally by CUDATensor and should not be called directly in most cases:

```python
# Internal usage (automatically handled)
from genesis.ndarray.cuda_memory_manager import allocate_memory, free_memory

# Allocate GPU memory
ptr = allocate_memory(nbytes)

# Free GPU memory  
free_memory(ptr)

# Get statistics
stats = memory_stats()
```

## Design Decisions

### Simplicity Over Optimization

The current implementation prioritizes:
- **Correctness**: All operations work reliably
- **Stability**: No complex edge cases or race conditions
- **Debuggability**: Simple, direct operations easy to trace

### Previous Complex Version (Deprecated)

An earlier version included advanced optimizations:
- Segment-block allocator with memory pooling
- Two-level caching (stream-local + global)
- Memory alignment and coalescing
- Complex statistics and performance monitoring

**Why it was removed**: The complex optimizations introduced precision errors in numerical computations, particularly in matrix multiplication operations with non-standard dimensions.

## Error Handling

The memory manager includes robust error handling:
- CUDA API errors are caught and reported with descriptive messages
- Memory leaks are prevented through proper destructor cleanup
- Graceful degradation when operations fail

## Performance Characteristics

While the current implementation is not optimized for speed, it provides:
- Consistent allocation/deallocation times
- No memory fragmentation issues
- Predictable memory usage patterns
- Zero precision errors in numerical operations

## Future Improvements

Potential optimizations that could be added while maintaining correctness:
- Memory pooling with careful precision testing
- Asynchronous allocation for large tensors
- Memory usage monitoring and reporting
- Integration with CUDA Memory Management APIs

## Troubleshooting

### Common Issues

1. **CUDA Context Errors**
   - Ensure CUDA drivers are properly installed
   - Check that GPU is accessible and not in use by other processes

2. **Memory Allocation Failures**
   - Monitor GPU memory usage with `nvidia-smi`
   - Reduce batch sizes or tensor dimensions if needed

3. **Performance Concerns**
   - Current implementation prioritizes correctness over speed
   - For performance-critical applications, consider optimizations carefully
   - Always test numerical precision after any memory management changes

### Debugging Tips

- Use `memory_stats()` to monitor allocation patterns
- Enable CUDA error checking in development builds
- Test with smaller tensors first to isolate memory vs computation issues

## Integration

The memory manager is tightly integrated with:
- **CUDATensor**: All tensor allocations go through this system
- **Triton Kernels**: GPU computations use allocated memory
- **Autograd System**: Gradient tensors use the same memory management