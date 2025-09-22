# 迁移指南：Genesis v1 到 v2

本指南帮助您将代码从Genesis v1.x迁移到Genesis v2.0，v2.0引入了重大的架构变更。

## 📋 概述

Genesis v2.0引入了重大的架构改进：
- **模块化后端系统**：分离了CPU和CUDA实现
- **统一设备抽象**：一致的设备管理
- **操作分发系统**：集中的操作路由
- **移除NDArray模块**：功能迁移到后端

## 🔄 破坏性变更总结

### 1. 移除NDArray模块
整个`genesis.ndarray`模块已被移除，其功能集成到新的后端系统中。

#### 之前 (v1.x)
```python
# 显式使用NDArray
from genesis.ndarray import NDArray
x = NDArray([1, 2, 3], device='cuda')
```

#### 之后 (v2.0)
```python
# 直接创建张量
import genesis
x = genesis.tensor([1, 2, 3], device='cuda')
```

### 2. 导入变更
由于重构，许多导入路径已更改。

#### 之前 (v1.x)
```python
from genesis.autograd import Tensor
from genesis.ndarray.device import Device
from genesis.ndarray.cuda_storage import CUDAStorage
```

#### 之后 (v2.0)
```python
from genesis import Tensor  # 或者直接使用genesis.tensor()
from genesis.device import Device
from genesis.backends.cuda import CUDAStorage
```

### 3. 后端访问
直接后端访问已被重构。

#### 之前 (v1.x)
```python
# 直接访问CUDA函数
from genesis.ndarray.cuda_backend import cuda_add
result = cuda_add(a, b)
```

#### 之后 (v2.0)
```python
# 操作自动分发到正确的后端
result = genesis.add(a, b)  # 如果张量在GPU上自动使用CUDA
```

## 🔧 代码迁移步骤

### 步骤1：更新导入
用新的导入替换旧的导入：

```python
# 旧导入 (v1.x) -> 新导入 (v2.0)
from genesis.autograd import Tensor          -> from genesis import tensor, Tensor
from genesis.ndarray import NDArray          -> # 移除，使用genesis.tensor()
from genesis.ndarray.device import Device    -> from genesis.device import Device
from genesis.backend import Backend          -> # 移除，自动处理
```

### 步骤2：替换NDArray使用
将NDArray操作转换为张量操作：

```python
# 之前 (v1.x)
def old_function():
    x = NDArray([1, 2, 3], device='cuda')
    y = NDArray([4, 5, 6], device='cuda')
    return x.add(y)

# 之后 (v2.0)
def new_function():
    x = genesis.tensor([1, 2, 3], device='cuda')
    y = genesis.tensor([4, 5, 6], device='cuda')
    return x + y  # 或者 genesis.add(x, y)
```

### 步骤3：更新设备处理
使用新的统一设备系统：

```python
# 之前 (v1.x)
from genesis.ndarray.device import get_device
device = get_device('cuda:0')

# 之后 (v2.0)
device = genesis.device('cuda:0')
```

### 步骤4：后端特定代码
如果您有后端特定代码，请更新它：

```python
# 之前 (v1.x) - 直接后端访问
from genesis.ndarray.cuda_backend import CUDABackend
backend = CUDABackend()
result = backend.matmul(a, b)

# 之后 (v2.0) - 使用操作分发
result = genesis.matmul(a, b)  # 自动路由到适当的后端
```

## 📝 常见迁移模式

### 模式1：张量创建
```python
# 之前 (v1.x)
def create_tensors_v1():
    x = NDArray([1, 2, 3])
    y = NDArray.zeros((3, 3))
    z = NDArray.ones((2, 2), device='cuda')
    return x, y, z

# 之后 (v2.0)
def create_tensors_v2():
    x = genesis.tensor([1, 2, 3])
    y = genesis.zeros((3, 3))
    z = genesis.ones((2, 2), device='cuda')
    return x, y, z
```

### 模式2：设备传输
```python
# 之前 (v1.x)
def transfer_v1(tensor):
    cuda_tensor = tensor.to_device('cuda')
    cpu_tensor = cuda_tensor.to_device('cpu')
    return cpu_tensor

# 之后 (v2.0)
def transfer_v2(tensor):
    cuda_tensor = tensor.to('cuda')
    cpu_tensor = cuda_tensor.to('cpu')
    return cpu_tensor
```

### 模式3：自定义操作
```python
# 之前 (v1.x) - 需要NDArray知识
def custom_op_v1(x):
    if x.device.is_cuda:
        result = cuda_custom_kernel(x.data)
    else:
        result = cpu_custom_impl(x.data)
    return NDArray(result, device=x.device)

# 之后 (v2.0) - 使用操作分发
def custom_op_v2(x):
    return genesis.ops.custom_operation(x)  # 自动分发
```

### 模式4：内存管理
```python
# 之前 (v1.x)
def manage_memory_v1():
    x = NDArray.zeros((1000, 1000), device='cuda')
    # 手动内存管理
    del x
    NDArray.cuda_empty_cache()

# 之后 (v2.0)
def manage_memory_v2():
    x = genesis.zeros((1000, 1000), device='cuda')
    # 改进的自动内存管理
    del x
    genesis.cuda.empty_cache()  # 仍可用但较少需要
```

## ⚠️ 潜在问题和解决方案

### 问题1：导入错误
**问题**：`ImportError: cannot import name 'NDArray'`

**解决方案**：用张量函数替换NDArray使用
```python
# 修复导入错误
# from genesis.ndarray import NDArray  # 移除这行
import genesis
x = genesis.tensor(data)  # 使用这个替代
```

### 问题2：设备属性错误
**问题**：`AttributeError: 'str' object has no attribute 'is_cuda'`

**解决方案**：使用适当的设备对象
```python
# 之前 - 设备有时是字符串
device = 'cuda'
if device == 'cuda':  # 字符串比较

# 之后 - 使用设备对象
device = genesis.device('cuda')
if device.is_cuda:  # 适当的属性
```

### 问题3：未找到后端方法
**问题**：直接后端方法调用失败

**解决方案**：使用操作分发系统
```python
# 之前 - 直接后端调用
result = backend.specific_operation(x)

# 之后 - 使用分发的操作
result = genesis.ops.specific_operation(x)
```

### 问题4：性能回退
**问题**：迁移后代码运行较慢

**解决方案**：
1. 确保张量在正确的设备上
2. 尽可能使用原地操作
3. 检查不必要的设备传输

```python
# 检查张量设备放置
print(f"张量设备：{x.device}")

# 使用原地操作
x.add_(y)  # 而不是 x = x + y

# 最小化设备传输
# 保持相关张量在同一设备上
```

## ✅ 迁移检查清单

使用此检查清单确保完整迁移：

- [ ] **移除NDArray导入**
  - [ ] 移除`from genesis.ndarray import NDArray`
  - [ ] 移除`from genesis.ndarray.device import Device`
  - [ ] 移除其他ndarray特定导入

- [ ] **更新张量创建**
  - [ ] 将`NDArray(data)`替换为`genesis.tensor(data)`
  - [ ] 将`NDArray.zeros()`替换为`genesis.zeros()`
  - [ ] 将`NDArray.ones()`替换为`genesis.ones()`

- [ ] **更新设备处理**
  - [ ] 使用`genesis.device()`创建设备
  - [ ] 更新设备属性访问
  - [ ] 检查设备传输方法

- [ ] **更新操作**
  - [ ] 用操作分发替换直接后端调用
  - [ ] 更新自定义操作实现
  - [ ] 验证操作行为一致性

- [ ] **测试功能**
  - [ ] 运行现有测试
  - [ ] 验证性能特征
  - [ ] 检查内存使用模式

## 🚀 利用新特性

### 增强的内存管理
```python
# 利用改进的内存池
genesis.cuda.set_memory_fraction(0.9)  # 使用90%的GPU内存

# 监控内存使用
stats = genesis.cuda.memory_stats()
print(f"内存效率：{stats.fragmentation_ratio:.2%}")
```

### 改进的设备管理
```python
# 使用自动设备选择
device = genesis.device('auto')  # 选择最佳可用设备

# 基于上下文的设备管理
with genesis.device('cuda:1'):
    x = genesis.randn(1000, 1000)  # 自动在cuda:1上
```

### 操作性能分析
```python
# 分析操作
with genesis.ops.profile() as prof:
    result = complex_computation(data)

prof.print_stats()  # 显示按操作的性能分解
```

## 🔗 其他资源

- [破坏性变更](breaking-changes.md) - 破坏性变更的完整列表
- [后端系统](../backends/index.md) - 了解新的后端架构
- [设备指南](../core-components/device.md) - v2.0中的设备管理
- [性能指南](../performance/optimization-guide.md) - 优化v2.0代码

## 💡 获取帮助

如果在迁移过程中遇到问题：

1. **先检查文档** - 大多数常见模式都有覆盖
2. **搜索问题** - 在GitHub问题中查找类似问题
3. **提问** - 使用"migration"标签创建新问题
4. **提供示例** - 包含前后代码片段

记住v2.0提供了更好的性能和更清洁的架构，因此迁移努力是值得的！