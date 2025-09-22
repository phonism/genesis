# Memory Management API

Genesis provides advanced CUDA memory management with reference-counted memory pools, comprehensive statistics, and performance optimization features.

## Device Memory Management

### Device Methods

#### `device.memory_allocated()`

Get the current memory allocated on the device.

**Returns:**
- int: Memory allocated in bytes

**Example:**
```python
import genesis

device = genesis.device('cuda')
allocated = device.memory_allocated()
print(f"Allocated: {allocated / 1e6:.1f} MB")
```

#### `device.memory_cached()`

Get the current memory cached by the allocator.

**Returns:**
- int: Memory cached in bytes

#### `device.memory_reserved()`

Get the total memory reserved by the allocator.

**Returns:**
- int: Memory reserved in bytes

#### `device.max_memory_allocated()`

Get the maximum memory allocated during the session.

**Returns:**
- int: Peak memory allocated in bytes

#### `device.max_memory_cached()`

Get the maximum memory cached during the session.

**Returns:**
- int: Peak memory cached in bytes

### Memory Statistics

#### `device.memory_stats()`

Get comprehensive memory usage statistics.

**Returns:**
- dict: Dictionary containing detailed memory statistics

**Statistics included:**
- `allocated_bytes`: Current allocated memory
- `cached_bytes`: Current cached memory  
- `reserved_bytes`: Total reserved memory
- `inactive_split_bytes`: Memory in inactive splits
- `active_bytes`: Memory in active use
- `cache_hit_rate`: Cache hit rate percentage
- `num_allocations`: Total number of allocations
- `num_cache_hits`: Number of cache hits
- `num_cache_misses`: Number of cache misses
- `peak_allocated`: Peak allocated memory
- `peak_cached`: Peak cached memory

**Example:**
```python
device = genesis.device('cuda')
stats = device.memory_stats()

print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Peak memory: {stats['peak_allocated'] / 1e9:.2f} GB")
print(f"Allocations: {stats['num_allocations']}")
```

#### `device.memory_summary()`

Get a human-readable memory usage summary.

**Returns:**
- str: Formatted memory usage summary

**Example:**
```python
device = genesis.device('cuda')
print(device.memory_summary())
```

### Memory Control

#### `device.empty_cache()`

Clear the memory cache, freeing cached but unused memory.

**Example:**
```python
device = genesis.device('cuda')
device.empty_cache()  # Free cached memory
```

#### `device.reset_memory_stats()`

Reset all memory statistics counters.

**Example:**
```python
device = genesis.device('cuda')
device.reset_memory_stats()
```

## Memory Profiling

### Context Manager

#### `genesis.profiler.profile_memory()`

Context manager for detailed memory profiling.

**Example:**
```python
import genesis

device = genesis.device('cuda')

with genesis.profiler.profile_memory() as prof:
    x = genesis.rand(4096, 4096, device=device)
    y = genesis.matmul(x, x.T)
    del x, y

# Get detailed memory usage report
print(prof.memory_summary())
print(f"Peak usage: {prof.peak_memory() / 1e6:.1f} MB")
```

### Memory Events

The memory manager tracks detailed allocation and deallocation events:

#### Allocation Events
- Timestamp
- Size allocated
- Memory address
- Duration of allocation
- Cache hit/miss status
- Thread ID

#### Deallocation Events  
- Timestamp
- Size deallocated
- Memory address
- Lifetime of allocation
- Thread ID

## Advanced Features

### Memory Pool Configuration

The memory manager uses different strategies for different allocation sizes:

- **Small allocations (<1MB)**: Reference-counted memory pool with caching
- **Large allocations (≥1MB)**: Direct CUDA allocation with segment management

### Cache Optimization

The memory pool automatically optimizes cache performance:

- **Warmup**: Pre-allocates common sizes
- **Hit Rate Tracking**: Monitors cache effectiveness
- **Adaptive Sizing**: Adjusts pool sizes based on usage patterns

### OOM Protection

The memory manager provides fast-fail OOM handling:

```python
try:
    # This might cause OOM
    huge_tensor = genesis.zeros(100000, 100000, device='cuda')
except RuntimeError as e:
    if "CUDA OOM" in str(e):
        print("Out of memory - consider reducing tensor size")
```

## Performance Monitoring

### Real-time Statistics

Monitor memory usage in real-time:

```python
import genesis
import time

device = genesis.device('cuda')

# Reset stats for clean measurement
device.reset_memory_stats()

# Perform operations
for i in range(100):
    x = genesis.rand(1000, 1000, device=device)
    y = x + x
    del x, y
    
    if i % 10 == 0:
        stats = device.memory_stats()
        print(f"Iteration {i}: Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

### Memory Efficiency Analysis

```python
device = genesis.device('cuda')

# Analyze memory efficiency
stats = device.memory_stats()
efficiency = stats['allocated_bytes'] / stats['reserved_bytes']
fragmentation = 1 - (stats['active_bytes'] / stats['allocated_bytes'])

print(f"Memory efficiency: {efficiency:.1%}")
print(f"Fragmentation: {fragmentation:.1%}")
```

## Best Practices

### Memory Optimization Tips

1. **Use appropriate tensor sizes**: Avoid extremely small or large tensors when possible
2. **Clear intermediate results**: Delete tensors that are no longer needed
3. **Monitor cache hit rates**: Aim for >95% hit rates for optimal performance  
4. **Use context managers**: For automatic cleanup in complex operations

### Example: Optimized Training Loop

```python
import genesis

device = genesis.device('cuda')
model = YourModel().to(device)
optimizer = genesis.optim.Adam(model.parameters())

# Reset stats to monitor training efficiency
device.reset_memory_stats()

for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch.to(device))
        loss = criterion(outputs, targets.to(device))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clear intermediate tensors
        del outputs, loss
        
        # Monitor memory every 100 batches
        if batch_idx % 100 == 0:
            stats = device.memory_stats()
            print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

## Troubleshooting

### Common Issues

1. **Low cache hit rate**: Caused by varying tensor sizes. Use consistent sizes when possible.
2. **Memory fragmentation**: Clear cache periodically with `device.empty_cache()`
3. **OOM errors**: Monitor peak memory usage and reduce batch sizes
4. **Memory leaks**: Use proper tensor deletion and avoid circular references

### Debugging Memory Issues

```python
import genesis

device = genesis.device('cuda')

# Enable detailed logging (if available)
genesis.set_memory_debug(True)

# Monitor memory throughout execution
def memory_checkpoint(name):
    stats = device.memory_stats()
    print(f"{name}: {stats['allocated_bytes'] / 1e6:.1f}MB allocated, "
          f"{stats['cache_hit_rate']:.1%} hit rate")

memory_checkpoint("Start")
x = genesis.rand(4096, 4096, device=device)
memory_checkpoint("After allocation")
del x
memory_checkpoint("After deletion")
```

## See Also

- [CUDA Storage](../core-components/cuda-storage.md) - CUDA-specific tensor operations
- [Performance Guide](../performance/optimization-guide.md) - Performance optimization strategies
- [Performance Tuning](../tutorials/performance-tuning.md) - Advanced profiling capabilities