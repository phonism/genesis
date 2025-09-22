# Genesis v2.0 破坏性变更

本文档提供Genesis v2.0中所有破坏性变更的完整列表。

## 🗑️ 移除的模块和类

### NDArray模块（完全移除）
**影响**：高 - 影响所有直接使用NDArray的代码

```python
# ❌ v2.0中已移除
from genesis.ndarray import NDArray, Device
from genesis.ndarray.cuda_storage import CUDAStorage
from genesis.ndarray.cpu_storage import CPUStorage
from genesis.ndarray.cuda_backend import CUDABackend

# ✅ v2.0替代方案
import genesis
from genesis.device import Device
from genesis.backends.cuda import CUDAStorage
from genesis.backends.cpu import CPUStorage
```

### Autograd模块重构
**影响**：中等 - 影响自定义autograd实现

```python
# ❌ 已移除的路径
from genesis.autograd import Variable, Context, Function
from genesis.autograd.engine import backward_engine

# ✅ 新路径
from genesis.tensor import Tensor  # Variable -> Tensor
from genesis.function import Function, Context
# backward_engine现在是内部的
```

## 🔄 API变更

### 张量创建
```python
# ❌ 旧方式不再有效
x = NDArray([1, 2, 3])
y = Variable([1, 2, 3], requires_grad=True)

# ✅ 新的统一API
x = genesis.tensor([1, 2, 3])
y = genesis.tensor([1, 2, 3], requires_grad=True)
```

### 设备规范
```python
# ❌ 旧设备处理
from genesis.ndarray.device import CUDADevice, CPUDevice
device = CUDADevice(0)
tensor = NDArray([1, 2, 3], device=device)

# ✅ 新设备系统
device = genesis.device('cuda:0')
tensor = genesis.tensor([1, 2, 3], device=device)
```

## 📦 导入路径变更

```python
# ❌ 旧导入
from genesis.autograd import Tensor
from genesis.ndarray import Device
from genesis.backend import get_current_backend

# ✅ 新导入
from genesis import Tensor, tensor  # 两者都可用
from genesis.device import Device
# 后端选择是自动的
```

## 🔧 配置变更

### 环境变量
```python
# ❌ 旧环境变量（不再使用）
GENESIS_NDARRAY_BACKEND=cuda
GENESIS_DEFAULT_DTYPE=float32

# ✅ 新环境变量
GENESIS_DEFAULT_DEVICE=cuda:0
GENESIS_CUDA_MEMORY_FRACTION=0.8
```

## 🔗 参见

- [迁移指南](v2-migration.md) - 详细的迁移步骤
- [后端系统](../backends/index.md) - 了解新架构