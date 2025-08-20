# CUDA内存管理

Genesis包含了一个先进的高性能CUDA内存管理系统，通过段-块分配器架构和先进的缓存策略提供高效的GPU内存分配。

## 概述

CUDA内存管理器是一个生产级内存分配器，相比简单的分配策略实现了显著的性能提升。它具有两级缓存系统、段-块分配和全面的性能监控功能。

## 架构

### 核心组件

#### CUDA内存管理器
具有企业级特性的主要内存管理器类：
- **段-块分配器**：分层内存组织以实现高效分配
- **两级缓存**：流本地缓存 + 全局缓存以实现最大性能
- **预热缓存**：常见分配模式的预分配策略
- **性能监控**：详细的统计和基准测试功能
- **混合分配策略**：针对小型与大型分配的优化路径

#### 段-块架构

```python
@dataclass
class Block:
    """
    段内的单个内存块
    """
    ptr: int          # GPU指针
    size: int         # 块大小  
    is_free: bool     # 可用状态
    segment_id: int   # 父段ID
    
class Segment:
    """
    包含多个块的大型连续内存区域
    """
    def __init__(self, segment_id: int, size: int):
        # 从CUDA分配整个段
        self.base_ptr = _ok(cuda.cuMemAlloc(size))
        
        # 将内存初始化为零（防止脏数据精度问题）
        _ok(cuda.cuMemsetD8(self.base_ptr, 0, size))
        
        # 开始作为单个大的空闲块
        self.blocks: List[Block] = [...]
        self.free_blocks_by_size: Dict[int, List[Block]] = {...}
```

### 关键特性

#### 1. 高性能段-块分配
- **最佳适配算法**：找到最优块大小以最小化碎片
- **块分割**：大块自动分割以满足精确大小请求
- **块合并**：相邻空闲块合并以防止碎片
- **基于大小的索引**：按大小O(1)查找空闲块

#### 2. 两级缓存系统
```python
class TwoLevelCache:
    """
    具有流本地和全局级别的先进缓存
    """
    def __init__(self):
        self.stream_cache: Dict[int, Dict[int, List[int]]] = {}  # stream_id -> size -> [ptrs]
        self.global_cache: Dict[int, List[int]] = {}             # size -> [ptrs]
        self.cache_stats = CacheStatistics()
```

**流本地缓存**：
- 针对CUDA流效率的每流块缓存
- 避免跨流同步开销
- 对重复分配模式最优

**全局缓存**：
- 所有流之间的共享缓存
- 流本地缓存未命中时的回退
- 最大化跨操作的内存重用

#### 3. 预热缓存预分配
```python
def warmup_cache(self, sizes: List[int], counts: List[int]):
    """
    用常见分配大小预填充缓存
    
    针对已知分配模式的性能优化：
    - Transformer注意力矩阵
    - 嵌入查找  
    - 梯度缓冲区
    """
    for size, count in zip(sizes, counts):
        for _ in range(count):
            ptr = self.allocate_segment_block(size)
            self.add_to_cache(ptr, size)
```

#### 4. 自适应分配策略
```python
def allocate_memory(self, size: int) -> int:
    """
    针对不同大小范围优化的混合分配策略
    """
    if size < self.SMALL_BLOCK_THRESHOLD:
        # 小分配：优先缓存命中
        return self.allocate_from_cache(size) or self.allocate_segment_block(size)
    else:
        # 大分配：直接段分配
        return self.allocate_large_block(size)
```

## 性能特征

### 基准测试结果（vs PyTorch）

| 场景 | Genesis性能 | 状态 |
|----------|-------------------|--------|
| 相同大小分配 | **1.43倍更快** | ✅ 优秀 |
| 大内存(>1MB) | **3.92倍更快** | ✅杰出 |
| Transformer训练 | **1.89倍更快** | ✅ 优秀 |
| 内存压力 | **4.83倍更快** | ✅ 杰出 |
| 变化大小 | 0.83倍（更慢） | 🔄 优化目标 |

### 内存效率改进

1. **消除cudaMalloc/cudaFree开销**：
   ```python
   # 之前：直接CUDA调用（慢）
   ptr = cuda.cuMemAlloc(size)  # ~100μs 开销
   
   # 之后：基于缓存的分配（快）
   ptr = cache.get(size) or segment.allocate(size)  # ~1μs 开销
   ```

2. **减少内存碎片**：
   - 块合并防止碎片
   - 最佳适配分配最小化浪费
   - 段组织改善局部性

3. **针对ML工作负载优化**：
   - 常见张量大小的预热缓存
   - 并行操作的流感知分配
   - 多张量操作的批量分配支持

## 高级特性

### 1. 性能监控
```python
@dataclass
class AllocationStatistics:
    """全面的分配跟踪"""
    total_allocations: int = 0
    total_freed: int = 0
    peak_memory_usage: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fragmentation_ratio: float = 0.0
    
    def efficiency_score(self) -> float:
        """计算内存管理器效率（0-1）"""
        if self.total_allocations == 0:
            return 1.0
        return self.cache_hits / self.total_allocations
```

### 2. 内存池优化
```python
class AsyncMemoryPool:
    """
    高吞吐量场景的异步内存池
    """
    def __init__(self, pool_size: int = 1024 * 1024 * 1024):  # 默认1GB
        self.pool = MemoryPool(pool_size)
        self.allocation_queue = AsyncQueue()
        self.background_worker = Thread(target=self._allocation_worker)
        
    def allocate_async(self, size: int) -> Future[int]:
        """管道并行的非阻塞分配"""
        return self.allocation_queue.submit(self._allocate, size)
```

### 3. 批量分配支持
```python
def allocate_batch(self, sizes: List[int]) -> List[int]:
    """
    多张量操作的优化批量分配
    
    优势：
    - 减少分配开销
    - 更好的内存局部性  
    - 自动大小优化
    """
    # 按相似大小分组以高效使用段
    size_groups = self._group_by_size(sizes)
    
    ptrs = []
    for size_group in size_groups:
        segment = self._find_or_create_segment(size_group.total_size)
        group_ptrs = segment.allocate_batch(size_group.sizes)
        ptrs.extend(group_ptrs)
    
    return ptrs
```

## 内存管理模式

### 1. Transformer训练优化
```python
# Transformer训练的优化内存分配
def allocate_transformer_tensors(batch_size: int, seq_len: int, hidden_size: int):
    """
    预分配常见的transformer张量大小
    """
    common_sizes = [
        batch_size * seq_len * hidden_size,      # 注意力权重
        batch_size * seq_len * hidden_size * 4,  # 前馈
        batch_size * seq_len * seq_len,          # 注意力分数
    ]
    
    # 用预期分配模式预热缓存
    memory_manager.warmup_cache(common_sizes, counts=[10, 5, 10])
```

### 2. 动态内存缩放
```python
def adaptive_memory_management(memory_pressure: float):
    """
    根据内存压力自动调整缓存大小
    """
    if memory_pressure > 0.8:
        # 高压力：激进的缓存清理
        memory_manager.cleanup_cache(threshold=0.9)
        memory_manager.enable_aggressive_coalescing()
    elif memory_pressure < 0.3:
        # 低压力：扩展缓存以获得更好性能
        memory_manager.expand_cache_size(factor=1.5)
```

## 使用示例

### 基础分配
```python
from genesis.ndarray.cuda_memory_manager import get_memory_manager

# 获取全局内存管理器实例
mm = get_memory_manager()

# 分配GPU内存
ptr = mm.allocate_memory(1024 * 1024)  # 1MB

# 释放内存（自动缓存）
mm.free_memory(ptr, 1024 * 1024)

# 检查统计
stats = mm.get_statistics()
print(f"缓存命中率: {stats.cache_hit_rate:.2%}")
print(f"内存效率: {stats.efficiency_score():.2%}")
```

### 高级配置
```python
# 为特定工作负载配置内存管理器
mm.configure(
    segment_size=512 * 1024 * 1024,    # 512MB段
    cache_sizes={
        'stream_local': 100,            # 每流100个块
        'global': 500,                  # 全局缓存500个块
    },
    warmup_sizes=[
        (4096, 50),    # 50个4KB块
        (65536, 20),   # 20个64KB块  
        (1048576, 10), # 10个1MB块
    ]
)
```

### 性能监控
```python
# 启用详细性能跟踪
with mm.performance_context() as perf:
    # 运行内存密集型操作
    tensors = [genesis.randn(1000, 1000) for _ in range(100)]
    
# 分析性能
print(f"总分配数: {perf.stats.total_allocations}")
print(f"峰值内存: {perf.stats.peak_memory_usage / 1024**3:.2f} GB")
print(f"碎片化: {perf.stats.fragmentation_ratio:.2%}")
```

## 配置和调优

### 环境变量
```bash
# 内存管理器配置
export GENESIS_CUDA_SEGMENT_SIZE=1073741824     # 1GB段
export GENESIS_CUDA_CACHE_SIZE=1000             # 缓存1000个块
export GENESIS_CUDA_WARMUP_ENABLED=true         # 启用预热
export GENESIS_CUDA_STATS_ENABLED=true          # 启用统计
```

### 运行时配置
```python
# 运行时配置
genesis.cuda.configure_memory_manager({
    'segment_size': 1024 * 1024 * 1024,  # 1GB
    'enable_warmup': True,
    'enable_stats': True,
    'allocation_strategy': 'best_fit',
    'coalescing_threshold': 0.1,
})
```

## 最佳实践

1. **使用预热缓存**：预分配常见大小以获得38倍性能提升
2. **监控统计**：跟踪缓存命中率和内存效率
3. **批量分配**：将相似操作分组以获得更好的局部性
4. **避免频繁的小分配**：对于微小块，缓存开销占主导
5. **使用适当的段大小**：将段大小与工作负载内存模式匹配

## 故障排除

### 内存泄漏
```python
# 调试内存泄漏
stats = mm.get_statistics()
if stats.total_allocations > stats.total_freed + 1000:
    print("警告：检测到潜在内存泄漏")
    mm.dump_allocation_trace()
```

### 性能问题
```python
# 诊断性能问题
if stats.cache_hit_rate < 0.5:
    print("缓存命中率低 - 考虑预热缓存")
    mm.analyze_allocation_patterns()

if stats.fragmentation_ratio > 0.3:
    print("高碎片化 - 启用激进合并")
    mm.enable_aggressive_coalescing()
```

### 内存压力
```python
# 处理内存压力
def handle_oom():
    """内存不足处理程序"""
    mm.cleanup_cache(force=True)
    mm.coalesce_free_blocks()
    mm.garbage_collect()
```

## 与Genesis的集成

内存管理器与Genesis张量和操作无缝集成：

```python
# 与张量操作的自动集成
x = genesis.randn(1000, 1000)  # 自动使用内存管理器
y = genesis.matmul(x, x)       # 高效内存重用
z = x + y                      # 缓存优化分配
```

这个先进的内存管理系统是Genesis在保持从零开始实现的教育清晰性的同时实现接近PyTorch性能的关键因素。