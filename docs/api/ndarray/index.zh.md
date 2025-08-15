# NDArray系统 (genesis.ndarray)

## 概述

`genesis.ndarray`模块提供低级张量操作和设备抽象层，为Genesis提供支持。它实现了双后端架构，为CPU和GPU执行提供了优化操作。

## 核心概念

### 双后端架构

Genesis使用独特的双后端方法：
- **CPU后端**: 利用PyTorch进行CPU操作，具有完全兼容性
- **GPU后端**: 纯CUDA/Triton实现，实现最大性能控制

### 设备抽象

所有计算通过`Device`抽象实现设备无关：
- 自动设备选择和内存管理
- CPU和GPU执行之间的无缝切换
- 优化的内存分配模式

### 性能优化

ndarray系统包括几个性能优化：
- **内核缓存**: 编译的Triton内核被缓存以供重用
- **自适应配置**: 块大小根据张量维度自动调整
- **内存视图**: 无需数据复制的高效张量视图
- **广播优化**: 元素级操作的智能广播

## 主要类

### `NDArray`

Genesis中的基础数组类型，提供设备无关的张量操作。

```python
class NDArray:
    """
    支持设备的N维数组。
    
    参数:
        data: 输入数据（numpy数组、列表或张量）
        device: 目标设备（cpu或cuda）
        dtype: 数组的数据类型
        
    属性:
        shape: 数组维度的元组
        dtype: 元素的数据类型
        device: 数组存储的设备
        data: 底层张量数据
    """
    
    def __init__(
        self, 
        data, 
        device: Optional[Device] = None, 
        dtype: Optional[DType] = None
    ):
```

#### 创建方法

```python
@staticmethod
def make(
    shape: Tuple[int, ...], 
    device: Optional[Device] = None, 
    dtype: DType = genesis.float32
) -> NDArray:
    """
    创建指定形状的未初始化数组。
    
    参数:
        shape: 数组的维度
        device: 目标设备
        dtype: 元素数据类型
        
    返回:
        新的NDArray实例
        
    示例:
        >>> arr = NDArray.make((10, 20), device=genesis.cuda(), dtype=genesis.float32)
        >>> print(arr.shape)  # (10, 20)
    """
```

#### 属性和方法

```python
@property
def shape(self) -> Tuple[int, ...]:
    """数组维度。"""

@property
def dtype(self) -> DType:
    """元素数据类型。"""

@property
def device(self) -> Device:
    """数组存储的设备。"""

def numel(self) -> int:
    """
    元素总数。
    
    返回:
        所有维度的乘积
        
    示例:
        >>> arr = NDArray.make((3, 4, 5))
        >>> print(arr.numel())  # 60
    """

def is_contiguous(self) -> bool:
    """
    检查数组是否具有连续的内存布局。
    
    返回:
        如果内存连续则为True
    """

def fill(self, value: float) -> None:
    """
    用常数值原地填充数组。
    
    参数:
        value: 填充值
        
    示例:
        >>> arr = NDArray.make((5, 5))
        >>> arr.fill(0.0)
        >>> # 数组现在包含全零
    """

def numpy(self) -> np.ndarray:
    """
    转换为NumPy数组。
    
    返回:
        复制数据的NumPy数组
        
    示例:
        >>> arr = NDArray([1, 2, 3], device=genesis.cuda())
        >>> np_arr = arr.numpy()  # 从GPU复制到CPU
    """

def cpu(self):
    """
    将数组传输到CPU。
    
    返回:
        数组数据的CPU版本
    """
```

### `Device`

支持CPU和CUDA执行的抽象设备接口。

```python
class Device:
    """
    计算后端的设备抽象。
    
    参数:
        name: 设备名称（'cpu'或'cuda'）
        mod: 操作的后端模块
        device_id: GPU设备索引（用于CUDA设备）
    """
    
    def __init__(
        self, 
        name: str, 
        mod: Any, 
        device_id: Optional[int] = None
    ):

    def enabled(self) -> bool:
        """
        检查设备是否可用。
        
        返回:
            如果设备可以使用则为True
        """
```

#### 张量创建

```python
def randn(
    self, 
    *shape: int, 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    从正态分布创建随机数组。
    
    参数:
        *shape: 数组维度
        dtype: 元素数据类型
        
    返回:
        具有随机值的NDArray
        
    示例:
        >>> device = genesis.cuda()
        >>> arr = device.randn(10, 10)  # 10x10随机数组
    """

def rand(
    self, 
    *shape: int, 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    从均匀分布[0, 1)创建随机数组。
    
    参数:
        *shape: 数组维度
        dtype: 元素数据类型
        
    返回:
        具有均匀随机值的NDArray
    """

def empty(
    self, 
    shape: Tuple[int, ...], 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    创建未初始化数组。
    
    参数:
        shape: 数组维度
        dtype: 元素数据类型
        
    返回:
        未初始化的NDArray
    """

def full(
    self, 
    shape: Tuple[int, ...], 
    fill_value: float, 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    创建用指定值填充的数组。
    
    参数:
        shape: 数组维度
        fill_value: 填充数组的值
        dtype: 元素数据类型
        
    返回:
        用fill_value填充的NDArray
        
    示例:
        >>> device = genesis.cpu()
        >>> ones = device.full((5, 5), 1.0)  # 5x5全1数组
    """

def one_hot(
    self, 
    n: int, 
    i: NDArray, 
    dtype: Optional[DType] = genesis.float32
) -> NDArray:
    """
    创建独热编码数组。
    
    参数:
        n: 类别数量
        i: 索引数组
        dtype: 元素数据类型
        
    返回:
        独热编码的NDArray
        
    示例:
        >>> device = genesis.cpu()
        >>> indices = NDArray([0, 2, 1], device=device)
        >>> one_hot = device.one_hot(3, indices)
        >>> # 形状: (3, 3) 独热编码
    """
```

## 设备函数

### 设备创建

```python
def cpu() -> Device:
    """
    创建CPU设备。
    
    返回:
        CPU设备实例
        
    示例:
        >>> cpu_device = genesis.cpu()
        >>> arr = NDArray([1, 2, 3], device=cpu_device)
    """

def cuda(index: int = 0) -> Device:
    """
    创建CUDA设备。
    
    参数:
        index: GPU设备索引
        
    返回:
        CUDA设备实例，如果CUDA不可用则为None
        
    示例:
        >>> gpu_device = genesis.cuda(0)  # 第一个GPU
        >>> if gpu_device.enabled():
        ...     arr = NDArray([1, 2, 3], device=gpu_device)
    """

def device(device_name: Union[str, int]) -> Device:
    """
    通过名称或索引创建设备。
    
    参数:
        device_name: 'cpu', 'cuda', 'cuda:N', 或GPU索引
        
    返回:
        设备实例
        
    示例:
        >>> dev1 = genesis.device('cuda:1')  # 第二个GPU
        >>> dev2 = genesis.device(1)         # 同上
        >>> dev3 = genesis.device('cpu')     # CPU设备
    """

def default_device() -> Device:
    """
    获取默认设备（CPU）。
    
    返回:
        默认设备实例
    """

def all_devices() -> List[Device]:
    """
    获取所有可用设备的列表。
    
    返回:
        设备实例列表
    """
```

## 操作

ndarray系统通过后端模块支持一套完整的操作。

### 算术操作

```python
# 二元操作
add(x, y)           # 元素级加法
sub(x, y)           # 元素级减法  
mul(x, y)           # 元素级乘法
truediv(x, y)       # 元素级除法
pow(x, scalar)      # 元素级幂运算

# 一元操作
log(x)              # 自然对数
exp(x)              # 指数
sin(x)              # 正弦
cos(x)              # 余弦
sqrt(x)             # 平方根
```

### 归约操作

```python
reduce_sum(x, axis=None, keepdims=False)    # 求和归约
reduce_max(x, axis=None, keepdims=False)    # 最大值归约
reduce_min(x, axis=None, keepdims=False)    # 最小值归约
```

### 比较操作

```python
maximum(x, y)       # 元素级最大值
minimum(x, y)       # 元素级最小值
```

### 矩阵操作

```python
matmul(x, y)        # 矩阵乘法
transpose(x, axes)  # 张量转置
```

## 性能优化

### 内核缓存

Triton内核自动缓存以供重用：

```python
from genesis.ndarray.kernel_cache import cached_kernel_call

# 内核按函数签名和参数缓存
cached_kernel_call(kernel_func, grid_func, *args, **kwargs)
```

### 自适应配置

块大小自动适应张量维度：

```python
from genesis.ndarray.adaptive_config import AdaptiveConfig

# 为张量形状获取优化配置
config = AdaptiveConfig.get_elementwise_config(shape)
block_size = config['BLOCK_SIZE']
grid = config['grid']
```

### 内存管理

高效的内存分配模式：
- 尽可能连续的内存布局
- 基于视图的操作避免复制
- 自动内存清理

## GPU后端（CUDA/Triton）

### GPU操作

纯CUDA/Triton实现，针对GPU操作进行性能优化。

### Triton内核

为最大性能手工优化的Triton内核：
- 具有广播的元素级操作
- 具有工作高效算法的归约操作
- 具有分块的矩阵乘法
- 内存优化访问模式

## 使用示例

### 基本数组创建

```python
import genesis

# 在不同设备上创建数组
cpu_arr = genesis.NDArray([1, 2, 3, 4], device=genesis.cpu())
gpu_arr = genesis.NDArray([1, 2, 3, 4], device=genesis.cuda())

# 创建特定形状
zeros = genesis.NDArray.make((100, 100), device=genesis.cuda())
zeros.fill(0.0)
```

### 设备操作

```python
# 随机数组
device = genesis.cuda(0)
random_normal = device.randn(1000, 1000)
random_uniform = device.rand(1000, 1000)

# 独热编码
indices = genesis.NDArray([0, 2, 1, 3], device=device)
one_hot = device.one_hot(4, indices)
```

### 内存传输

```python
# GPU到CPU传输
gpu_data = genesis.NDArray([1, 2, 3], device=genesis.cuda())
cpu_data = gpu_data.cpu()
numpy_data = gpu_data.numpy()

# CPU到GPU传输  
cpu_data = genesis.NDArray([1, 2, 3], device=genesis.cpu())
gpu_data = genesis.NDArray(cpu_data, device=genesis.cuda())
```

### 性能监控

```python
import time

# 计时操作
start = time.time()
result = genesis.ndarray.add(x, y)
end = time.time()
print(f"操作用时 {(end - start) * 1000:.2f}ms")
```

## 后端选择

Genesis自动选择合适的后端：

```python
# CPU操作使用PyTorch后端
cpu_device = genesis.cpu()
x = genesis.NDArray([1, 2, 3], device=cpu_device)

# GPU操作使用Triton/CUDA后端
gpu_device = genesis.cuda()
if gpu_device.enabled():
    x = genesis.NDArray([1, 2, 3], device=gpu_device)
    # 使用优化的Triton内核
```

## 错误处理

ndarray系统提供全面的错误处理：

```python
try:
    # 尝试GPU操作
    gpu_device = genesis.cuda()
    if not gpu_device.enabled():
        raise RuntimeError("CUDA不可用")
    
    arr = genesis.NDArray([1, 2, 3], device=gpu_device)
except RuntimeError as e:
    # 回退到CPU
    print(f"GPU错误: {e}，使用CPU")
    cpu_device = genesis.cpu()
    arr = genesis.NDArray([1, 2, 3], device=cpu_device)
```

## 最佳实践

1. **设备选择**: 使用前检查设备可用性
2. **内存管理**: 谨慎地在设备间传输  
3. **批量操作**: 尽可能同时处理多个张量
4. **连续内存**: 确保数组连续以获得最佳性能
5. **错误处理**: 始终优雅地处理CUDA可用性

## 性能提示

1. **使用适当的块大小** 用于GPU内核
2. **最小化设备传输** 在CPU和GPU之间
3. **利用内核缓存** 通过重用相似操作
4. **使用视图而不是副本** 尽可能
5. **批量相似操作** 摊销内核启动开销

## 另请参阅

- [张量操作](../autograd.md) - 高级张量接口
- [神经网络模块](../nn/modules.md) - 在ndarray基础上构建
- [性能指南](../../performance/) - 优化技术
- [CUDA存储](../../../core-components/cuda-storage.md) - 低级CUDA实现