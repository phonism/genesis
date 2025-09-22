# 内存管理 API

Genesis 提供高级的 CUDA 内存管理功能，包括引用计数内存池、综合统计信息和性能优化特性。

## 设备内存管理

### 设备方法

#### `device.memory_allocated()`

获取设备上当前分配的内存。

**返回值:**
- int: 已分配的内存字节数

**示例:**
```python
import genesis

device = genesis.device('cuda')
allocated = device.memory_allocated()
print(f"已分配: {allocated / 1e6:.1f} MB")
```

#### `device.memory_cached()`

获取分配器当前缓存的内存。

**返回值:**
- int: 已缓存的内存字节数

#### `device.memory_reserved()`

获取分配器保留的总内存。

**返回值:**
- int: 已保留的内存字节数

#### `device.max_memory_allocated()`

获取会话期间分配的最大内存。

**返回值:**
- int: 峰值已分配内存字节数

#### `device.max_memory_cached()`

获取会话期间缓存的最大内存。

**返回值:**
- int: 峰值已缓存内存字节数

### 内存统计

#### `device.memory_stats()`

获取全面的内存使用统计信息。

**返回值:**
- dict: 包含详细内存统计信息的字典

**包含的统计信息:**
- `allocated_bytes`: 当前已分配内存
- `cached_bytes`: 当前已缓存内存  
- `reserved_bytes`: 总保留内存
- `inactive_split_bytes`: 非活动分片中的内存
- `active_bytes`: 活动使用中的内存
- `cache_hit_rate`: 缓存命中率百分比
- `num_allocations`: 分配总次数
- `num_cache_hits`: 缓存命中次数
- `num_cache_misses`: 缓存未命中次数
- `peak_allocated`: 峰值已分配内存
- `peak_cached`: 峰值已缓存内存

**示例:**
```python
device = genesis.device('cuda')
stats = device.memory_stats()

print(f"缓存命中率: {stats['cache_hit_rate']:.1%}")
print(f"峰值内存: {stats['peak_allocated'] / 1e9:.2f} GB")
print(f"分配次数: {stats['num_allocations']}")
```

#### `device.memory_summary()`

获取人类可读的内存使用摘要。

**返回值:**
- str: 格式化的内存使用摘要

**示例:**
```python
device = genesis.device('cuda')
print(device.memory_summary())
```

### 内存控制

#### `device.empty_cache()`

清空内存缓存，释放已缓存但未使用的内存。

**示例:**
```python
device = genesis.device('cuda')
device.empty_cache()  # 释放缓存内存
```

#### `device.reset_memory_stats()`

重置所有内存统计计数器。

**示例:**
```python
device = genesis.device('cuda')
device.reset_memory_stats()
```

## 内存性能分析

### 上下文管理器

#### `genesis.profiler.profile_memory()`

用于详细内存分析的上下文管理器。

**示例:**
```python
import genesis

device = genesis.device('cuda')

with genesis.profiler.profile_memory() as prof:
    x = genesis.rand(4096, 4096, device=device)
    y = genesis.matmul(x, x.T)
    del x, y

# 获取详细的内存使用报告
print(prof.memory_summary())
print(f"峰值使用: {prof.peak_memory() / 1e6:.1f} MB")
```

### 内存事件

内存管理器跟踪详细的分配和释放事件：

#### 分配事件
- 时间戳
- 分配大小
- 内存地址
- 分配持续时间
- 缓存命中/未命中状态
- 线程 ID

#### 释放事件  
- 时间戳
- 释放大小
- 内存地址
- 分配生命周期
- 线程 ID

## 高级功能

### 内存池配置

内存管理器对不同分配大小使用不同策略：

- **小分配 (<1MB)**: 使用缓存的引用计数内存池
- **大分配 (≥1MB)**: 使用段管理的直接 CUDA 分配

### 缓存优化

内存池自动优化缓存性能：

- **预热**: 预分配常见大小
- **命中率跟踪**: 监控缓存效果
- **自适应大小**: 基于使用模式调整池大小

### OOM 保护

内存管理器提供快速失败的 OOM 处理：

```python
try:
    # 这可能导致 OOM
    huge_tensor = genesis.zeros(100000, 100000, device='cuda')
except RuntimeError as e:
    if "CUDA OOM" in str(e):
        print("内存不足 - 请考虑减小张量大小")
```

## 性能监控

### 实时统计

实时监控内存使用：

```python
import genesis
import time

device = genesis.device('cuda')

# 重置统计以便干净测量
device.reset_memory_stats()

# 执行操作
for i in range(100):
    x = genesis.rand(1000, 1000, device=device)
    y = x + x
    del x, y
    
    if i % 10 == 0:
        stats = device.memory_stats()
        print(f"迭代 {i}: 缓存命中率: {stats['cache_hit_rate']:.1%}")
```

### 内存效率分析

```python
device = genesis.device('cuda')

# 分析内存效率
stats = device.memory_stats()
efficiency = stats['allocated_bytes'] / stats['reserved_bytes']
fragmentation = 1 - (stats['active_bytes'] / stats['allocated_bytes'])

print(f"内存效率: {efficiency:.1%}")
print(f"碎片化: {fragmentation:.1%}")
```

## 最佳实践

### 内存优化技巧

1. **使用合适的张量大小**: 尽量避免过小或过大的张量
2. **清理中间结果**: 删除不再需要的张量
3. **监控缓存命中率**: 追求 >95% 的命中率以获得最佳性能  
4. **使用上下文管理器**: 在复杂操作中自动清理

### 示例: 优化的训练循环

```python
import genesis

device = genesis.device('cuda')
model = YourModel().to(device)
optimizer = genesis.optim.Adam(model.parameters())

# 重置统计以监控训练效率
device.reset_memory_stats()

for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播
        outputs = model(batch.to(device))
        loss = criterion(outputs, targets.to(device))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 清理中间张量
        del outputs, loss
        
        # 每 100 个批次监控内存
        if batch_idx % 100 == 0:
            stats = device.memory_stats()
            print(f"缓存命中率: {stats['cache_hit_rate']:.1%}")
```

## 故障排除

### 常见问题

1. **低缓存命中率**: 由张量大小变化引起。尽可能使用一致的大小。
2. **内存碎片化**: 定期使用 `device.empty_cache()` 清理缓存
3. **OOM 错误**: 监控峰值内存使用并减少批大小
4. **内存泄漏**: 使用适当的张量删除，避免循环引用

### 调试内存问题

```python
import genesis

device = genesis.device('cuda')

# 启用详细日志记录（如果可用）
genesis.set_memory_debug(True)

# 在整个执行过程中监控内存
def memory_checkpoint(name):
    stats = device.memory_stats()
    print(f"{name}: {stats['allocated_bytes'] / 1e6:.1f}MB 已分配, "
          f"{stats['cache_hit_rate']:.1%} 命中率")

memory_checkpoint("开始")
x = genesis.rand(4096, 4096, device=device)
memory_checkpoint("分配后")
del x
memory_checkpoint("删除后")
```

## 另请参阅

- [CUDA 存储](../core-components/cuda-storage.md) - CUDA 特定的张量操作
- [性能指南](../performance/optimization-guide.md) - 性能优化策略
- [性能调优](../tutorials/performance-tuning.md) - 高级性能分析功能