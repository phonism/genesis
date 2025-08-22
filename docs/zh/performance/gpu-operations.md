# GPU操作性能指南

本指南涵盖Genesis中GPU操作的优化，重点介绍模块化GPU操作结构、Triton内核实现和性能调优策略。

## 概述

Genesis实现了复杂的GPU后端，包括：
- 使用Triton的模块化GPU操作
- 自定义CUDA内存管理
- 自适应块大小优化
- 性能监控和分析工具

## 架构概述

### 模块化GPU操作结构

Genesis将GPU操作分离为专门的模块：

```
genesis/ndarray/gpu_ops/
├── __init__.py          # 操作注册和分派
├── basic_ops.py         # 逐元素操作（add、mul等）
├── tensor_ops.py        # 张量操作（matmul、conv等）  
├── random_ops.py        # 随机数生成
└── reduction_ops.py     # 约简操作（sum、mean等）
```

### 操作分派系统

```python
# genesis/ndarray/gpu_ops/__init__.py
from .basic_ops import add_triton, mul_triton, div_triton
from .tensor_ops import matmul_triton, conv2d_triton  
from .reduction_ops import sum_triton, mean_triton

# 动态分派的操作注册表
GPU_OPS_REGISTRY = {
    'add': add_triton,
    'mul': mul_triton,
    'div': div_triton,
    'matmul': matmul_triton,
    'sum': sum_triton,
    'mean': mean_triton,
}

def dispatch_gpu_op(op_name, *args, **kwargs):
    """将操作分派到相应的GPU内核。"""
    if op_name not in GPU_OPS_REGISTRY:
        raise NotImplementedError(f"GPU操作 {op_name} 未实现")
    
    return GPU_OPS_REGISTRY[op_name](*args, **kwargs)
```

## Triton内核实现

### 基本逐元素操作

```python
# genesis/ndarray/gpu_ops/basic_ops.py
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """优化的逐元素加法内核。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 使用向量化加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # 计算
    output = x + y
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

def add_triton(x, y):
    """基于Triton的张量加法。"""
    output = genesis.empty_like(x)
    n_elements = x.numel()
    
    # 基于张量大小的自适应块大小
    if n_elements < 262144:  # < 256K元素
        BLOCK_SIZE = 256
    elif n_elements < 4194304:  # < 4M元素  
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](
        x.data_ptr(), y.data_ptr(), output.data_ptr(),
        n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
```

### 高级张量操作

```python
# genesis/ndarray/gpu_ops/tensor_ops.py
@triton.jit  
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """高性能矩阵乘法内核。"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K))
        
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_triton(a, b):
    """使用Triton优化的矩阵乘法。"""
    assert a.shape[-1] == b.shape[-2], f"形状不匹配: {a.shape} @ {b.shape}"
    
    M, K = a.shape[-2:]
    K2, N = b.shape[-2:]
    assert K == K2
    
    c = genesis.empty((*a.shape[:-2], M, N), dtype=a.dtype, device=a.device)
    
    # 基于问题大小优化块大小
    if M >= 2048 and N >= 2048:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32
    elif M >= 512 and N >= 512:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 32, 32, 32
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a.data_ptr(), b.data_ptr(), c.data_ptr(),
        M, N, K,
        a.stride(-2), a.stride(-1),
        b.stride(-2), b.stride(-1),
        c.stride(-2), c.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c
```

### 内存优化的约简操作

```python
# genesis/ndarray/gpu_ops/reduction_ops.py
@triton.jit
def sum_kernel(
    input_ptr, output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """内存高效的求和内核。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 按块加载和求和
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(x)
    
    # 使用原子加法进行最终约简
    tl.atomic_add(output_ptr, block_sum)

@triton.jit
def reduce_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr  
):
    """具有最优内存访问的2D约简内核。"""
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    offs_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    mask_x = offs_x < n_rows
    mask_y = offs_y < n_cols
    
    # 加载块
    ptrs = input_ptr + offs_x[:, None] * n_cols + offs_y[None, :]
    mask = mask_x[:, None] & mask_y[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0)
    
    # 块内约简
    result = tl.sum(x, axis=1)  # 跨列求和
    
    # 存储结果
    out_ptrs = output_ptr + offs_x
    tl.store(out_ptrs, result, mask=mask_x)

def sum_triton(x, dim=None, keepdim=False):
    """优化的张量求和。"""
    if dim is None:
        # 全局求和
        result = genesis.zeros((), dtype=x.dtype, device=x.device)
        n_elements = x.numel()
        
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        sum_kernel[grid](
            x.data_ptr(), result.data_ptr(),
            n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return result
    
    else:
        # 特定维度约简
        # 实现特定维度约简
        return reduce_along_dim(x, dim, keepdim)
```

## 性能优化策略

### 1. 自适应块大小优化

```python
class AdaptiveBlockSize:
    """基于张量特性动态优化块大小。"""
    
    def __init__(self):
        self.cache = {}
        self.performance_history = {}
    
    def get_optimal_block_size(self, operation, tensor_size, dtype):
        """获取给定操作和张量的最优块大小。"""
        cache_key = (operation, tensor_size, dtype.name)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 基于张量大小和操作确定块大小
        if operation == 'elementwise':
            if tensor_size < 262144:  # < 256K元素
                block_size = 256
            elif tensor_size < 4194304:  # < 4M元素
                block_size = 512  
            else:
                block_size = 1024
                
        elif operation == 'matmul':
            # 矩阵乘法特定优化
            sqrt_size = int(tensor_size ** 0.5)
            if sqrt_size < 512:
                block_size = (32, 32, 32)
            elif sqrt_size < 2048:
                block_size = (64, 64, 32)
            else:
                block_size = (128, 128, 32)
                
        elif operation == 'reduction':
            # 约简操作优化
            block_size = min(1024, triton.next_power_of_2(tensor_size))
        
        else:
            # 默认回退
            block_size = 512
        
        self.cache[cache_key] = block_size
        return block_size
    
    def update_performance(self, operation, tensor_size, dtype, block_size, elapsed_time):
        """更新性能历史记录用于未来优化。"""
        key = (operation, tensor_size, dtype.name, block_size)
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append(elapsed_time)
        
        # 只保留最近的测量
        if len(self.performance_history[key]) > 10:
            self.performance_history[key] = self.performance_history[key][-10:]

# 全局优化器实例
block_optimizer = AdaptiveBlockSize()
```

### 2. 内存访问模式优化

```python
@triton.jit
def coalesced_copy_kernel(
    src_ptr, dst_ptr,
    n_elements, stride_src, stride_dst,
    BLOCK_SIZE: tl.constexpr
):
    """内存合并的张量拷贝内核。"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # 确保合并的内存访问
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 使用适当的步长加载
    src_offsets = offsets * stride_src
    dst_offsets = offsets * stride_dst
    
    data = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + dst_offsets, data, mask=mask)

@triton.jit  
def transpose_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """缓存友好的矩阵转置。"""
    pid = tl.program_id(axis=0)
    
    # 基于tile的转置以获得更好的缓存使用
    row_start = (pid // (n_cols // BLOCK_SIZE)) * BLOCK_SIZE
    col_start = (pid % (n_cols // BLOCK_SIZE)) * BLOCK_SIZE
    
    rows = row_start + tl.arange(0, BLOCK_SIZE)
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    
    # 加载tile
    input_offsets = rows[:, None] * n_cols + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    
    tile = tl.load(input_ptr + input_offsets, mask=mask)
    
    # 存储转置的tile
    output_offsets = cols[:, None] * n_rows + rows[None, :]
    tl.store(output_ptr + output_offsets, tl.trans(tile), mask=tl.trans(mask))
```

### 3. 内核融合优化

```python
@triton.jit
def fused_linear_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    n_batch, n_input, n_output,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """融合的线性层 + ReLU内核以减少内存带宽。"""
    pid = tl.program_id(axis=0)
    
    # 矩阵乘法逻辑（简化）
    # ... matmul computation ...
    
    # 融合的ReLU激活
    result = tl.maximum(matmul_result + bias, 0.0)
    
    # 单次内存写入
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    seq_len, head_dim,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """融合的注意力计算内核。"""
    # 计算注意力分数
    scores = compute_qk_scores(q_ptr, k_ptr, seq_len, head_dim)
    
    # 应用缩放和softmax
    scores = scores * scale
    attention_weights = tl_softmax(scores, axis=-1)
    
    # 将注意力应用到值
    output = compute_attention_output(attention_weights, v_ptr, seq_len, head_dim)
    
    # 单次输出写入
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_linear_relu(x, weight, bias):
    """融合的线性层与ReLU激活。"""
    batch_size, input_size = x.shape
    output_size = weight.shape[0]
    
    output = genesis.empty(batch_size, output_size, dtype=x.dtype, device=x.device)
    
    # 融合的最优块大小
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  
    BLOCK_SIZE_K = 32
    
    grid = lambda meta: (
        triton.cdiv(batch_size, meta['BLOCK_SIZE_M']) * 
        triton.cdiv(output_size, meta['BLOCK_SIZE_N']),
    )
    
    fused_linear_relu_kernel[grid](
        x.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
        batch_size, input_size, output_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output
```

## 性能监控和分析

### 1. 内置性能指标

```python
import time
import contextlib

class GPUProfiler:
    """分析GPU操作性能。"""
    
    def __init__(self):
        self.metrics = {}
        self.current_operation = None
    
    @contextlib.contextmanager
    def profile_operation(self, operation_name):
        """用于分析操作的上下文管理器。"""
        self.current_operation = operation_name
        
        # 计时前同步
        genesis.cuda.synchronize()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            genesis.cuda.synchronize()
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            
            self.metrics[operation_name].append(elapsed)
    
    def get_stats(self, operation_name=None):
        """获取性能统计。"""
        if operation_name:
            times = self.metrics.get(operation_name, [])
            if not times:
                return None
            
            return {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'count': len(times)
            }
        else:
            stats = {}
            for op_name in self.metrics:
                stats[op_name] = self.get_stats(op_name)
            return stats
    
    def print_summary(self):
        """打印性能摘要。"""
        print("GPU操作性能摘要:")
        print("-" * 50)
        
        for op_name, stats in self.get_stats().items():
            print(f"{op_name}:")
            print(f"  平均: {stats['mean']*1000:.3f}ms")
            print(f"  最小:  {stats['min']*1000:.3f}ms")
            print(f"  最大:  {stats['max']*1000:.3f}ms")
            print(f"  次数: {stats['count']}")
            print()

# 全局分析器实例
gpu_profiler = GPUProfiler()

# 在操作中使用分析器的示例
def add_with_profiling(x, y):
    with gpu_profiler.profile_operation('add'):
        return add_triton(x, y)
```

### 2. 内存带宽分析

```python
def analyze_memory_bandwidth(operation_func, tensor_sizes, dtype=genesis.float32):
    """分析操作的内存带宽利用率。"""
    
    results = []
    theoretical_bandwidth = 1555e9  # A800 HBM2e带宽，字节/秒
    
    for size in tensor_sizes:
        # 创建测试张量
        if isinstance(size, tuple):
            x = genesis.randn(*size, dtype=dtype, device='cuda')
            y = genesis.randn(*size, dtype=dtype, device='cuda')
        else:
            x = genesis.randn(size, dtype=dtype, device='cuda')
            y = genesis.randn(size, dtype=dtype, device='cuda')
        
        # 计算理论传输的字节数
        bytes_per_element = dtype.itemsize
        total_elements = x.numel()
        
        # 对于二元操作：读取2个张量 + 写入1个张量
        total_bytes = total_elements * bytes_per_element * 3
        
        # 预热
        for _ in range(5):
            _ = operation_func(x, y)
        
        # 计时操作
        genesis.cuda.synchronize()
        start_time = time.perf_counter()
        
        num_iterations = 10
        for _ in range(num_iterations):
            result = operation_func(x, y)
        
        genesis.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 计算指标
        elapsed_time = (end_time - start_time) / num_iterations
        achieved_bandwidth = total_bytes / elapsed_time
        bandwidth_efficiency = achieved_bandwidth / theoretical_bandwidth
        
        results.append({
            'size': size,
            'elements': total_elements,
            'elapsed_ms': elapsed_time * 1000,
            'bandwidth_gb_s': achieved_bandwidth / 1e9,
            'efficiency_percent': bandwidth_efficiency * 100,
            'theoretical_gb_s': theoretical_bandwidth / 1e9
        })
        
        print(f"大小 {size}: {achieved_bandwidth/1e9:.1f} GB/s ({bandwidth_efficiency:.1%})")
    
    return results

# 分析加法操作性能
sizes = [(256, 256), (1024, 1024), (2048, 2048), (4096, 4096)]
bandwidth_results = analyze_memory_bandwidth(add_triton, sizes)
```

### 3. 自动性能调优

```python
class AutoTuner:
    """自动调优内核参数以获得最佳性能。"""
    
    def __init__(self):
        self.best_configs = {}
    
    def tune_kernel(self, kernel_func, test_inputs, param_space):
        """自动调优内核参数。"""
        best_time = float('inf')
        best_config = None
        
        print(f"正在调优内核，共有 {len(param_space)} 个配置...")
        
        for i, config in enumerate(param_space):
            try:
                # 预热
                for _ in range(3):
                    _ = kernel_func(*test_inputs, **config)
                
                # 计时执行
                genesis.cuda.synchronize()
                start_time = time.perf_counter()
                
                num_runs = 10
                for _ in range(num_runs):
                    result = kernel_func(*test_inputs, **config)
                
                genesis.cuda.synchronize()
                end_time = time.perf_counter()
                
                elapsed = (end_time - start_time) / num_runs
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_config = config
                
                print(f"配置 {i+1}: {elapsed*1000:.3f}ms - {config}")
                
            except Exception as e:
                print(f"配置 {i+1} 失败: {e}")
        
        print(f"最佳配置: {best_config} ({best_time*1000:.3f}ms)")
        return best_config, best_time

# 矩阵乘法自动调优示例
def tune_matmul():
    # 定义参数空间
    block_sizes = [
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64},
    ]
    
    # 测试输入
    a = genesis.randn(1024, 1024, device='cuda')
    b = genesis.randn(1024, 1024, device='cuda')
    
    # 运行自动调优器
    tuner = AutoTuner()
    best_config, best_time = tuner.tune_kernel(
        matmul_triton, [a, b], block_sizes
    )
    
    return best_config
```

## 最佳实践

### 1. 内核开发指南

- **内存合并**: 确保连续的内存访问模式
- **块大小优化**: 使用2的幂，考虑占用率
- **寄存器使用**: 监控大内核的寄存器溢出
- **共享内存**: 为数据重用使用共享内存
- **分支发散最小化**: 可能时避免条件分支

### 2. 性能优化检查清单

- [ ] 分析内存带宽利用率
- [ ] 为目标GPU优化块大小
- [ ] 最小化内核启动开销
- [ ] 为相关操作使用内核融合
- [ ] 监控GPU占用率和资源使用
- [ ] 优化后验证数值精度

### 3. 调试GPU操作

```python
def debug_gpu_operation(operation_func, *inputs):
    """使用详细分析调试GPU操作。"""
    
    print("GPU操作调试信息:")
    print("=" * 40)
    
    # 输入分析
    for i, inp in enumerate(inputs):
        print(f"输入 {i}:")
        print(f"  形状: {inp.shape}")
        print(f"  数据类型: {inp.dtype}")
        print(f"  设备: {inp.device}")
        print(f"  内存使用: {inp.numel() * inp.dtype.itemsize / 1e6:.1f} MB")
        print(f"  连续性: {inp.is_contiguous()}")
        print()
    
    # GPU内存状态
    print("GPU内存状态:")
    print(f"  已分配: {genesis.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"  缓存: {genesis.cuda.memory_cached() / 1e6:.1f} MB")
    print()
    
    # 执行带分析的操作
    genesis.cuda.synchronize()
    start_time = time.perf_counter()
    
    result = operation_func(*inputs)
    
    genesis.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 结果分析
    print("操作结果:")
    print(f"  执行时间: {(end_time - start_time) * 1000:.3f}ms")
    print(f"  输出形状: {result.shape}")
    print(f"  输出数据类型: {result.dtype}")
    print(f"  输出设备: {result.device}")
    print()
    
    # 数值验证
    print("数值验证:")
    print(f"  最小值: {result.min().item():.6f}")
    print(f"  最大值: {result.max().item():.6f}")
    print(f"  平均值: {result.mean().item():.6f}")
    print(f"  存在NaN: {genesis.isnan(result).any().item()}")
    print(f"  存在Inf: {genesis.isinf(result).any().item()}")
    
    return result

# 使用示例
x = genesis.randn(1000, 1000, device='cuda')
y = genesis.randn(1000, 1000, device='cuda')
result = debug_gpu_operation(add_triton, x, y)
```

这个全面的指南涵盖了Genesis中模块化GPU操作架构，提供了详细的实现示例和优化策略，以实现最佳性能。