# CUDA存储系统

Genesis的CUDA存储（CUDAStorage）是框架的核心组件，提供纯CUDA实现的GPU内存管理和操作，完全独立于PyTorch，直接使用CUDA Python API。

## 🎯 设计目标

### 独立性
- **纯CUDA实现**：不依赖PyTorch的GPU后端
- **直接内存管理**：使用CUDA Python API直接管理GPU内存
- **高性能**：针对GPU优化的内存访问模式

### 兼容性  
- **PyTorch风格API**：保持与PyTorch张量的接口兼容性
- **自动微分支持**：与Genesis自动微分系统无缝集成
- **类型安全**：完整的类型注解和运行时检查

## 🏗️ 架构设计

### IndexPlan架构

CUDATensor使用先进的IndexPlan架构来处理复杂的张量索引操作：

```python
class IndexKind(Enum):
    VIEW = "view"           # 纯视图操作，零拷贝
    GATHER = "gather"       # 收集操作，用于高级索引  
    SCATTER = "scatter"     # 散布操作，用于赋值
    COPY = "copy"          # 步长拷贝
    FILL = "fill"          # 填充操作

@dataclass
class IndexPlan:
    """统一的索引计划"""
    kind: IndexKind
    result_shape: Optional[Tuple[int, ...]] = None
    result_strides: Optional[Tuple[int, ...]] = None
    ptr_offset_bytes: int = 0
    index_tensor: Optional['CUDATensor'] = None
    needs_mask_compaction: bool = False
    temp_memory_bytes: int = 0
```

### 内存管理

```python
class AsyncMemoryPool:
    """异步内存池，优化GPU内存分配性能"""
    
    def __init__(self):
        self.free_blocks = {}  # 按大小组织的空闲块
        self.allocated_blocks = {}  # 已分配的块
        self.alignment = 512  # 内存对齐，与PyTorch一致
        
    def allocate(self, size_bytes: int) -> int:
        """分配对齐的GPU内存"""
        
    def deallocate(self, ptr: int):
        """释放GPU内存到池中重用"""
```

## 💡 核心特性

### 1. 高效的索引操作

```python
import genesis

# 创建CUDA张量
x = genesis.randn(1000, 1000, device='cuda')

# 基础索引 - 使用VIEW操作，零拷贝
y = x[10:20, 50:100]  # IndexPlan.kind = VIEW

# 高级索引 - 使用GATHER操作  
indices = genesis.tensor([1, 3, 5, 7], device='cuda')
z = x[indices]  # IndexPlan.kind = GATHER

# 布尔索引 - 自动优化
mask = x > 0.5
w = x[mask]  # 根据稠密度选择最优策略
```

### 2. 内存高效的操作

```python
# 就地操作，避免内存分配
x = genesis.randn(1000, 1000, device='cuda')
x += 1.0  # 就地加法

# 视图操作，零拷贝
y = x.view(100, 10000)  # 改变形状但不复制数据
z = x.transpose(0, 1)   # 转置视图

# 步长操作，高效实现
w = x[::2, ::3]  # 步长索引，使用优化的COPY操作
```

### 3. Triton内核集成

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """优化的Triton加法内核"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

# CUDATensor自动调用优化的Triton内核
def add_cuda_tensor(x: CUDATensor, y: CUDATensor) -> CUDATensor:
    """CUDA张量加法，使用Triton优化"""
    output = CUDATensor(x.shape, x.dtype)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](x.data_ptr(), y.data_ptr(), output.data_ptr(), 
                     n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
```

## 🚀 基础使用

### 创建张量

```python
import genesis

# 从数据创建
data = [[1.0, 2.0], [3.0, 4.0]]
tensor = genesis.tensor(data, device='cuda')

# 直接创建特定形状
zeros = genesis.zeros(100, 100, device='cuda')
ones = genesis.ones(50, 50, device='cuda')  
random = genesis.randn(200, 200, device='cuda')

# 指定数据类型
float16_tensor = genesis.randn(100, 100, dtype=genesis.float16, device='cuda')
int_tensor = genesis.randint(0, 10, (50, 50), device='cuda')

print(f"张量形状: {tensor.shape}")
print(f"数据类型: {tensor.dtype}")
print(f"设备: {tensor.device}")
print(f"步长: {tensor.strides}")
```

### 基础操作

```python
# 数学运算
x = genesis.randn(100, 100, device='cuda')
y = genesis.randn(100, 100, device='cuda')

# 逐元素运算
z = x + y      # 加法
w = x * y      # 乘法  
u = x.pow(2)   # 幂运算
v = x.exp()    # 指数函数

# 归约操作
sum_all = x.sum()           # 全局求和
sum_dim = x.sum(dim=0)      # 按维度求和
mean_val = x.mean()         # 平均值
max_val, indices = x.max(dim=1)  # 最大值和索引

# 线性代数
a = genesis.randn(100, 50, device='cuda')
b = genesis.randn(50, 200, device='cuda') 
c = genesis.matmul(a, b)    # 矩阵乘法

# 形状操作
reshaped = x.view(10, 1000)        # 改变形状
transposed = x.transpose(0, 1)     # 转置  
flattened = x.flatten()            # 展平
```

### 高级索引

```python
# 创建测试张量
data = genesis.arange(0, 100, device='cuda').view(10, 10)
print("原始数据:")
print(data)

# 基础切片
slice_basic = data[2:5, 3:7]  # 行2-4，列3-6
print("基础切片:", slice_basic.shape)

# 步长索引
slice_stride = data[::2, 1::2]  # 每隔一行，从第1列开始每隔一列
print("步长索引:", slice_stride.shape)

# 高级索引 - 整数数组
row_indices = genesis.tensor([0, 2, 4, 6], device='cuda')
col_indices = genesis.tensor([1, 3, 5, 7], device='cuda')
advanced = data[row_indices, col_indices]  # 选择特定位置
print("高级索引结果:", advanced)

# 布尔索引
mask = data > 50
masked_data = data[mask]  # 选择大于50的元素
print("布尔索引结果:", masked_data)

# 混合索引
mixed = data[row_indices, 2:8]  # 特定行的列范围
print("混合索引:", mixed.shape)
```

## 🔧 内存管理

### 内存池优化

```python
# 查看内存使用情况
print(f"已分配内存: {genesis.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"缓存内存: {genesis.cuda.memory_cached() / 1024**2:.1f} MB")

# 手动内存管理
x = genesis.randn(1000, 1000, device='cuda')
print(f"创建张量后: {genesis.cuda.memory_allocated() / 1024**2:.1f} MB")

del x  # 删除引用
genesis.cuda.empty_cache()  # 清空缓存
print(f"清理后: {genesis.cuda.memory_allocated() / 1024**2:.1f} MB")

# 内存快照（调试用）
snapshot = genesis.cuda.memory_snapshot()
for entry in snapshot[:3]:  # 显示前3个条目
    print(f"地址: {entry['address']}, 大小: {entry['size']} bytes")
```

### 异步操作

```python
# 异步内存操作
with genesis.cuda.stream():
    x = genesis.randn(1000, 1000, device='cuda')
    y = genesis.randn(1000, 1000, device='cuda')
    z = genesis.matmul(x, y)  # 异步执行
    
    # 其他CPU工作可以并行进行
    print("矩阵乘法正在GPU上异步执行...")
    
    # 同步等待结果  
    genesis.cuda.synchronize()
    print("计算完成:", z.shape)
```

## ⚡ 性能优化

### 1. 内存访问模式优化

```python
def inefficient_access():
    """低效的内存访问模式"""
    x = genesis.randn(1000, 1000, device='cuda')
    result = genesis.zeros(1000, device='cuda')
    
    # 非连续访问，缓存未命中
    for i in range(1000):
        result[i] = x[i, ::10].sum()  # 步长访问
    
    return result

def efficient_access():  
    """高效的内存访问模式"""
    x = genesis.randn(1000, 1000, device='cuda')
    
    # 连续访问，充分利用缓存
    indices = genesis.arange(0, 1000, 10, device='cuda')
    selected = x[:, indices]  # 批量选择
    result = selected.sum(dim=1)  # 向量化求和
    
    return result

# 性能对比
import time

start = time.time()
result1 = inefficient_access()
time1 = time.time() - start

start = time.time()  
result2 = efficient_access()
time2 = time.time() - start

print(f"低效方法: {time1:.4f}s")
print(f"高效方法: {time2:.4f}s")  
print(f"加速比: {time1/time2:.2f}x")
```

### 2. 批量操作优化

```python
def batch_operations_demo():
    """展示批量操作的性能优势"""
    
    # 创建测试数据
    matrices = [genesis.randn(100, 100, device='cuda') for _ in range(10)]
    
    # 方法1: 逐个处理（低效）
    start = time.time()
    results1 = []
    for matrix in matrices:
        result = matrix.exp().sum()
        results1.append(result)
    time1 = time.time() - start
    
    # 方法2: 批量处理（高效）
    start = time.time()
    batched = genesis.stack(matrices, dim=0)  # [10, 100, 100]
    results2 = batched.exp().sum(dim=(1, 2))  # [10]
    time2 = time.time() - start
    
    print(f"逐个处理: {time1:.4f}s")
    print(f"批量处理: {time2:.4f}s")
    print(f"加速比: {time1/time2:.2f}x")

batch_operations_demo()
```

### 3. 就地操作

```python
def inplace_operations_demo():
    """展示就地操作的内存效率"""
    
    # 非就地操作（创建新张量）
    x = genesis.randn(1000, 1000, device='cuda')
    start_memory = genesis.cuda.memory_allocated()
    
    y = x + 1.0      # 创建新张量
    z = y * 2.0      # 再创建新张量
    w = z.exp()      # 又创建新张量
    
    memory_after = genesis.cuda.memory_allocated()
    print(f"非就地操作内存增长: {(memory_after - start_memory) / 1024**2:.1f} MB")
    
    # 就地操作（修改原张量）
    x = genesis.randn(1000, 1000, device='cuda')
    start_memory = genesis.cuda.memory_allocated()
    
    x += 1.0         # 就地加法
    x *= 2.0         # 就地乘法  
    x.exp_()         # 就地指数函数
    
    memory_after = genesis.cuda.memory_allocated()
    print(f"就地操作内存增长: {(memory_after - start_memory) / 1024**2:.1f} MB")

inplace_operations_demo()
```

## 🐛 调试和诊断

### 内存泄漏检测

```python
def detect_memory_leaks():
    """检测内存泄漏"""
    genesis.cuda.empty_cache()
    initial_memory = genesis.cuda.memory_allocated()
    
    # 执行一些操作
    for i in range(100):
        x = genesis.randn(100, 100, device='cuda')
        y = x.matmul(x)
        del x, y
    
    genesis.cuda.empty_cache()
    final_memory = genesis.cuda.memory_allocated()
    
    if final_memory > initial_memory:
        print(f"可能存在内存泄漏: {(final_memory - initial_memory) / 1024**2:.1f} MB")
    else:
        print("未检测到内存泄漏")

detect_memory_leaks()
```

### 错误诊断

```python
def diagnose_cuda_errors():
    """CUDA错误诊断"""
    try:
        # 可能出错的操作
        x = genesis.randn(1000, 1000, device='cuda')
        y = genesis.randn(500, 500, device='cuda')  # 形状不匹配
        z = genesis.matmul(x, y)
        
    except RuntimeError as e:
        print(f"CUDA错误: {e}")
        
        # 检查CUDA状态
        if genesis.cuda.is_available():
            print(f"CUDA设备: {genesis.cuda.get_device_name()}")
            print(f"CUDA能力: {genesis.cuda.get_device_capability()}")
            print(f"可用内存: {genesis.cuda.get_device_properties().total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA不可用")

diagnose_cuda_errors()
```

## 🔄 与PyTorch互操作

```python
import torch

def pytorch_interop_demo():
    """展示与PyTorch的互操作性"""
    
    # Genesis张量转PyTorch
    genesis_tensor = genesis.randn(100, 100, device='cuda')
    
    # 转换为PyTorch（共享内存）
    pytorch_tensor = torch.as_tensor(genesis_tensor.detach().cpu().numpy()).cuda()
    
    print(f"Genesis形状: {genesis_tensor.shape}")
    print(f"PyTorch形状: {pytorch_tensor.shape}")
    
    # PyTorch张量转Genesis  
    torch_data = torch.randn(50, 50, device='cuda')
    genesis_from_torch = genesis.tensor(torch_data.cpu().numpy(), device='cuda')
    
    print(f"转换成功，Genesis张量: {genesis_from_torch.shape}")

pytorch_interop_demo()
```

## 📊 性能基准

```python
def benchmark_cuda_tensor():
    """CUDA张量性能基准测试"""
    
    sizes = [100, 500, 1000, 2000]
    
    print("矩阵乘法性能对比 (Genesis vs PyTorch):")
    print("-" * 50)
    
    for size in sizes:
        # Genesis测试
        x_gen = genesis.randn(size, size, device='cuda')
        y_gen = genesis.randn(size, size, device='cuda')
        
        genesis.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z_gen = genesis.matmul(x_gen, y_gen)
        genesis.cuda.synchronize()
        genesis_time = (time.time() - start) / 10
        
        # PyTorch测试
        x_torch = torch.randn(size, size, device='cuda')
        y_torch = torch.randn(size, size, device='cuda')
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z_torch = torch.matmul(x_torch, y_torch)
        torch.cuda.synchronize() 
        pytorch_time = (time.time() - start) / 10
        
        ratio = genesis_time / pytorch_time
        print(f"{size}x{size}: Genesis {genesis_time:.4f}s, PyTorch {pytorch_time:.4f}s, 比率 {ratio:.2f}")

benchmark_cuda_tensor()
```

## 🎯 最佳实践

### 1. 内存管理最佳实践

```python
# ✅ 好的做法
def good_memory_practice():
    with genesis.cuda.device(0):  # 明确指定设备
        x = genesis.randn(1000, 1000, device='cuda')
        
        # 使用就地操作
        x += 1.0
        x *= 0.5
        
        # 及时释放大张量
        del x
        genesis.cuda.empty_cache()

# ❌ 避免的做法  
def bad_memory_practice():
    tensors = []
    for i in range(100):
        x = genesis.randn(1000, 1000, device='cuda')
        y = x + 1.0  # 创建额外副本
        tensors.append(y)  # 保持所有引用，内存无法释放
    # 内存会快速耗尽
```

### 2. 性能优化最佳实践

```python
# ✅ 向量化操作
def vectorized_operations():
    x = genesis.randn(1000, 1000, device='cuda')
    
    # 使用向量化函数
    result = genesis.relu(x).sum(dim=1).mean()
    
# ❌ 避免循环
def avoid_loops():
    x = genesis.randn(1000, 1000, device='cuda')
    
    # 避免Python循环
    result = 0
    for i in range(1000):
        result += x[i].sum()  # 每次都启动CUDA kernel
```

### 3. 调试最佳实践

```python
# 启用CUDA错误检查
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 使用断言检查张量属性
def safe_tensor_operation(x, y):
    assert x.device == y.device, "张量必须在同一设备上"
    assert x.shape == y.shape, f"形状不匹配: {x.shape} vs {y.shape}"
    
    return x + y
```

## ❓ 常见问题

### Q: CUDA内存不足怎么办？
A: 
```python
# 减小批量大小
batch_size = 32  # 改为16或8

# 使用梯度累积
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

# 启用混合精度
x = genesis.randn(1000, 1000, dtype=genesis.float16, device='cuda')

# 定期清理内存
genesis.cuda.empty_cache()
```

### Q: 为什么CUDA操作很慢？  
A: 检查以下几点：
```python
# 1. 确保张量在GPU上
assert x.device.type == 'cuda'

# 2. 避免频繁的CPU-GPU传输
# 错误做法
for i in range(1000):
    cpu_data = x.cpu().numpy()  # 每次都传输

# 正确做法
cpu_data = x.cpu().numpy()  # 只传输一次

# 3. 使用适当的数据类型
x = genesis.randn(1000, 1000, dtype=genesis.float16, device='cuda')  # 更快
```

### Q: 如何调试CUDA kernel错误？
A:
```python
# 1. 启用同步错误检查
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 2. 检查tensor有效性
def check_tensor(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name}包含NaN"
    assert not torch.isinf(tensor).any(), f"{name}包含Inf"
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")

# 3. 使用CUDA调试工具
# cuda-memcheck python your_script.py
# compute-sanitizer python your_script.py
```

---

!!! tip "性能提示"
    CUDA张量的性能很大程度上取决于内存访问模式和批量操作的使用。优先考虑向量化操作和合理的内存布局。

**准备深入了解更多吗？**

[下一步：张量操作优化](tensor-operations.zh.md){ .md-button .md-button--primary }
[返回核心组件](index.zh.md){ .md-button }